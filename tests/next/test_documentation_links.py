"""Validate dashboard help against real MkDocs output at both supported bases.

The documentation dependencies are optional for analysis-only installations.
With the ``docs`` dependency group installed, this builds the actual theme and
Markdown extensions instead of reproducing their heading/URL generation rules.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit


REPOSITORY = Path(__file__).resolve().parents[2]
HELP_LINKS = REPOSITORY / "dashboard/src/lib/help-links.json"
DOCS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("mkdocs", "material"))


class _RenderedPage(HTMLParser):
    """Collect browser-facing targets, heading IDs, and Material search config."""

    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.canonicals = []
        self.config_chunks = []
        self.in_config = False
        self.dashboard_div_depth = 0
        self.feed(path.read_text(encoding="utf-8"))
        self.close()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "div" and (self.dashboard_div_depth or attributes.get("id") == "dashboard-return"):
            self.dashboard_div_depth += 1
        if tag == "script" and attributes.get("id") == "__config":
            self.in_config = True
        if tag == "link" and "canonical" in attributes.get("rel", "").split():
            self.canonicals.append(attributes.get("href", ""))
        target_attribute = {"a": "href", "link": "href", "script": "src", "img": "src"}.get(tag)
        if target_attribute and attributes.get(target_attribute):
            # This explicit application shortcut intentionally leaves /docs/.
            # Its container is hidden for standalone documentation hosting.
            if not (tag == "a" and self.dashboard_div_depth):
                self.links.append(attributes[target_attribute])

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_config = False
        if tag == "div" and self.dashboard_div_depth:
            self.dashboard_div_depth -= 1

    def handle_data(self, data):
        if self.in_config:
            self.config_chunks.append(data)

    @property
    def material_config(self):
        return json.loads("".join(self.config_chunks))


@unittest.skipUnless(DOCS_AVAILABLE, "Install the docs dependency group to validate rendered documentation links.")
class DocumentationLinkContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="wm-next-doc-links-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.sites = []
        for prefix in ("docs", "wm-states"):
            site_url = f"https://docs.example.test/{prefix}/"
            directory = Path(cls.temporary.name) / prefix
            environment = {**os.environ, "MKDOCS_SITE_URL": site_url}
            result = subprocess.run(
                [sys.executable, "-m", "mkdocs", "build", "--strict", "--config-file",
                 str(REPOSITORY / "mkdocs.yml"), "--site-dir", str(directory)],
                cwd=REPOSITORY, env=environment, text=True, capture_output=True, timeout=60,
            )
            if result.returncode:
                raise AssertionError(f"MkDocs build failed for {site_url}:\n{result.stdout}\n{result.stderr}")
            pages = {path.resolve(): _RenderedPage(path) for path in directory.rglob("*.html")}
            cls.sites.append((directory.resolve(), site_url, pages))

    @staticmethod
    def page_url(directory, site_url, page):
        relative = page.relative_to(directory).as_posix()
        if relative.endswith("index.html"):
            relative = relative[:-len("index.html")]
        return urljoin(site_url, relative)

    def assert_local_target(self, directory, site_url, pages, source_url, reference):
        """Resolve as a browser would, then check the generated file and anchor."""
        resolved = urlsplit(urljoin(source_url, reference))
        origin = urlsplit(site_url)
        if resolved.scheme not in ("http", "https") or resolved.netloc != origin.netloc:
            return
        self.assertTrue(
            resolved.path.startswith(origin.path) or resolved.path == origin.path.rstrip("/"),
            f"Documentation link escaped its host prefix: {source_url} -> {reference}",
        )
        relative = unquote(resolved.path[len(origin.path):])
        target = (directory / relative).resolve()
        self.assertTrue(target.is_relative_to(directory), f"Link escaped generated site: {reference}")
        if target.is_dir():
            target /= "index.html"
        self.assertTrue(target.is_file(), f"Missing target: {source_url} -> {reference} ({target})")
        if resolved.fragment and target.suffix == ".html":
            self.assertIn(
                unquote(resolved.fragment), pages[target].ids,
                f"Missing anchor: {source_url} -> {reference}",
            )

    def test_dashboard_help_covers_every_stage_and_resolves_real_anchors(self):
        from scripts.next.pipeline import STAGES

        manifest = json.loads(HELP_LINKS.read_text(encoding="utf-8"))
        stage_ids = [stage["id"] for stage in manifest["stageHelp"]]
        self.assertEqual(len(stage_ids), len(set(stage_ids)), "Duplicate dashboard help stage IDs")
        self.assertEqual(set(stage_ids), set(STAGES), "Every pipeline stage needs contextual help")
        references = []

        def collect_paths(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "path":
                        references.append(child)
                    else:
                        collect_paths(child)
            elif isinstance(value, list):
                for child in value:
                    collect_paths(child)

        collect_paths(manifest)
        references.extend(manifest["fieldLinks"].values())
        references.extend(f"next/troubleshooting/#{entry['anchor']}" for entry in manifest["troubleshooting"])
        self.assertTrue(references, "Help must contain documentation targets")
        for directory, site_url, pages in self.sites:
            for reference in references:
                with self.subTest(site_url=site_url, help_target=reference):
                    self.assertIsInstance(reference, str)
                    self.assertFalse(urlsplit(reference).scheme or reference.startswith("/"),
                                     "Help targets must remain relative to the configured docs base")
                    self.assert_local_target(directory, site_url, pages, site_url, reference)

    def test_navigation_and_assets_stay_within_each_documentation_base(self):
        for directory, site_url, pages in self.sites:
            self.assertTrue(pages)
            for page, parsed in pages.items():
                source_url = self.page_url(directory, site_url, page)
                for reference in parsed.links:
                    with self.subTest(page=source_url, link=reference):
                        self.assert_local_target(directory, site_url, pages, source_url, reference)

    def test_canonical_urls_use_configured_host_and_prefix(self):
        for directory, site_url, pages in self.sites:
            for page, parsed in pages.items():
                if page.name == "404.html":
                    continue  # MkDocs intentionally omits a canonical URL for errors.
                with self.subTest(site_url=site_url, page=str(page.relative_to(directory))):
                    self.assertEqual(parsed.canonicals, [self.page_url(directory, site_url, page)])
                    self.assertNotIn("localhost", parsed.canonicals[0])
                    self.assertNotIn("127.0.0.1", parsed.canonicals[0])

    def test_material_search_worker_index_and_result_locations_resolve(self):
        for directory, site_url, pages in self.sites:
            index = directory / "search/search_index.json"
            self.assertTrue(index.is_file())
            search = json.loads(index.read_text(encoding="utf-8"))
            self.assertTrue(search["docs"], "Documentation search index must not be empty")
            for page, parsed in pages.items():
                source_url = self.page_url(directory, site_url, page)
                config = parsed.material_config
                with self.subTest(page=source_url):
                    self.assert_local_target(directory, site_url, pages, source_url, config["search"])
                    # Material loads the index from its page-relative site base.
                    base_url = urljoin(source_url, config["base"].rstrip("/") + "/")
                    self.assertEqual(base_url, site_url)
                    self.assert_local_target(directory, site_url, pages, base_url, "search/search_index.json")
            for entry in search["docs"]:
                with self.subTest(site_url=site_url, search_location=entry["location"]):
                    self.assert_local_target(directory, site_url, pages, site_url, entry["location"])


if __name__ == "__main__":
    unittest.main()
