# Dashboard

The local dashboard configures and runs the complete `next` pipeline, shows
processing status, and brings session results and saved figures into one viewer.
It uses the same Python stages, JSON settings, caches, and manifest history as
the command-line runner.

## First-time setup

Open a terminal in the repository root: the directory containing `README.md`,
`pyproject.toml`, and `dashboard/`. Run every command on this page from there.
You need `uv`, Python 3.12, and Node.js **22.12 or newer** with npm. Node.js 24
is used for validation.

Check your Node version with `node --version`. If you already use nvm, you can
install and select Node.js 24 with:

```bash
nvm install 24
nvm use 24
```

Install the Python and browser dependencies, then build and start the dashboard
with its documentation:

```bash
uv sync --python 3.12 --group dashboard --group docs --locked
npm --prefix dashboard ci
uv run --group dashboard --group docs --locked python -m scripts.next.dashboard --build
```

`npm ci` installs the versions in the lockfile. `--build` builds the React
interface and MkDocs guide, then starts one Python server for both. It installs
frontend dependencies if `node_modules/` is missing; run `npm ci` yourself when
the lockfile changes. Builds must succeed before the server starts.
You can open the dashboard without recordings to explore the interface or view
existing runs. To launch an analysis, first
[prepare the recordings](getting-started.md#prepare-the-recordings).

When the terminal says the server is running, open
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser.
The server does not open a browser automatically. An empty run library on a
fresh checkout is normal; generated data and results are not included in Git.

## Open the dashboard again

On later visits, open a terminal in the repository root and run just:

```bash
uv run --group dashboard --group docs --locked python -m scripts.next.dashboard
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). If the server is
already running in another terminal, simply open that address.

- **Keep the server terminal open** while using the dashboard.
- Closing or refreshing the browser tab leaves analysis running.
- **Ctrl+C in the server terminal stops the dashboard and cancels any analysis
  it launched that is still running.** Saved outputs remain on disk.
- After frontend or documentation changes, stop the server when no analysis is
  running and repeat the start command with `--build`, then refresh the browser.
  The [development workflow](#frontend-development) also supports rebuilding
  assets while the server remains running.
- After backend Python changes, stop and restart the server when no analysis
  is running. The normal launcher does not reload Python code automatically.

### Use another port

If port 8000 is already in use, start on a free port:

```bash
uv run --group dashboard --group docs --locked python -m scripts.next.dashboard --port 8001
```

Open [http://127.0.0.1:8001/](http://127.0.0.1:8001/) for that server. The guide
moves with it to `/docs/`; no extra documentation server is needed.

| Service | Usual address | When to use it |
| --- | --- | --- |
| Dashboard | `http://127.0.0.1:8000/` | Configure runs, follow progress, and inspect results |
| Pipeline guide | `http://127.0.0.1:8000/docs/` | Read or search the full guide from the dashboard |
| Developer API reference | `http://127.0.0.1:8000/api/docs` | Inspect REST endpoints |
| MkDocs development preview | `http://127.0.0.1:8001/` | Edit the guide with automatic reload |
| Vite development server | `http://127.0.0.1:5173/` | Edit the frontend with automatic browser updates |

Only development previews need separate ports. If you moved the dashboard to
8001, choose another free port for a MkDocs preview. Include both
`--group dashboard --group docs` in `uv` commands when sharing one environment
so it retains both dependency groups.

The dashboard is intended for one trusted user on their own computer. Its
launcher binds to loopback addresses and it has no account or authentication
system. It reads Python pickle caches on the server: only place caches you trust
in this repository's `cache/` directory. Do not expose the service through a
public reverse proxy.

## Get help without leaving your work

Click **Help** in the top bar to open a side panel. It starts with guidance for
your current workspace and provides searchable topics for setup, configuration,
live progress, results, comparisons, and analysis methods. These concise tips
are available even when the documentation has not yet been built. On a smaller
screen, the panel uses the available width. Close it with its close button or
Escape to return to the same workspace.

- **Pipeline guide** in the sidebar opens the full searchable MkDocs guide.
- Each configuration stage has a **Stage methods** link to its method section.
- **Learn more** links explain consequential settings such as screening checks,
  null-shuffle time structure, and state thresholds.
- Expanded analysis topics include optional background reference links to
  official library documentation. The full methods guide also cites papers and
  explains which rules are specific to this pipeline.
- Error messages include troubleshooting links where guidance is available.
- **API reference** is the separate developer reference, now at `/api/docs`.

Documentation and reference links open in a new tab, leaving your unsaved settings and
live progress intact. The local guide has a **Back to dashboard** shortcut;
switch to your original tab to resume its exact view. Search within the help
panel searches its topics; use the guide's search for the complete documentation.

The guide is built from this checkout, so it can describe the same code and
presets you are running. Rebuild it after updating the repository. It can also
be published separately, including on GitHub Pages; see
[standalone hosting](../development.md#publish-the-guide-on-github-pages).

## Try your first run

1. Prepare the four local example sessions using
   [Getting started](getting-started.md#prepare-the-recordings).
2. Open **Configure pipeline**, choose the **Smoke test** preset, and click
   **Use example data** to set the data directory to `data/example`.
3. Choose a fresh output directory, such as `cache/next_dashboard_001`, and a
   worker count suitable for your computer. Two workers is a modest starting
   point. Keep all eleven stages selected for an end-to-end check.
4. Click **Validate & preview**, review the generated command, and start the
   run. Follow the stages and log in the processing view.
5. Open the **Run library**, select the run, and choose a session to inspect
   its figures and metrics. Use the comparison workspace to place two sessions
   or two run directories side by side.

The smoke preset reduces computation to check that the pipeline works. Use
**Example pipeline** for the example scientific analysis; see
[Analysis methods](methods.md) for its choices. If you already have results,
you can go straight to the run library without starting a new analysis.

## Configure an analysis

The new-run form starts with the choices in
`configs/next/example_pipeline.json`; settings omitted from that preset use the
corresponding Python stage defaults. The dashboard selects all eleven stages
initially. The command-line runner still selects its first five stages unless
`--stages` is specified. See [Analysis methods](methods.md) for the example
preset's scientific choices and [Pipeline stages](pipeline.md) for dependencies.

1. Set the data directory containing the session `.mat` files and a new run
   directory, such as `cache/next_dashboard_001`.
2. Choose the stages and worker count. Stages keep their pipeline order;
   prerequisites are not automatically added. A partial run needs the required
   upstream caches in the same run directory.
3. Review each stage's arguments. Fields are generated from the actual Python
   configuration classes, including explicit screening switches, null-shuffle
   counts, state thresholds, and mixed-effects settings. Use the JSON editor for
   direct edits to the per-stage settings. Explicit `null` keeps an optional
   setting unset; deleting a field restores its Python default. Switching back
   to the form applies valid JSON and keeps invalid drafts available to correct.
4. Validate the configuration, review the generated command, and start the run.

Dropdowns use the same controls throughout the dashboard. Focus one and press
Enter or Space to open it. With the menu open, use the up/down arrows to move,
Home/End to reach the first/last option, or type the start of an option's label
to jump to it. Enter confirms; Escape closes the menu without changing the
selection and returns focus to the control. Menu descriptions show full paths
or identifying metadata where names need context. Hover over a selected control
to see its full label and description if the displayed text is shortened.

Search within a stage by parameter name or description; spaces and underscores
both work. **Changed** shows only parameters that differ from the example
pipeline. The badges and stage counts compare effective values: the example
JSON plus Python defaults for omitted fields. They always use this reference,
including when you start from the smoke preset or copied settings. **Use example**
restores one parameter without resetting the rest of your setup. Changing which
stage you are viewing does not change whether that stage is included in the run.

Validation catches unknown settings, incorrect types, and configuration errors
that can be checked before execution. Loading session data, checking cache
provenance, and fitting models still happen in the pipeline; a valid form does
not guarantee that an analysis will succeed.

Screening diagnostics have a separate JSON file. In the `select` settings,
enable `save_extended_diagnostics` and keep or change the
`diagnostics_figure_config` path. Edit its session/cell targets and plot settings
in that file; the dashboard's stage JSON editor stores the path. The supplied
file targets available sessions and caps plots at 12 cells each. Set the path
to `null` for CSV-only diagnostics. These settings do not restrict screening
or diagnostic CSV rows. See [screening diagnostics](configuration.md#screening-diagnostics)
for the schema, template, and full-session trace behavior.

Use the smoke preset for an integration check, with the four sessions prepared
in [Getting started](getting-started.md). It reduces decoding and model work;
it is not a substitute for the example preset's analysis settings. Selecting a
preset replaces the stage settings while keeping your run details and stage
selection. Review the shared paths and worker counts before starting.

To start from a previous invocation, choose **Reuse settings** in the run
library. This copies its resolved analysis settings and shared run arguments
into a new form with a fresh output directory. It does not copy existing output
files. A copied partial invocation still requires upstream results: choose the
required stages for a new run, or deliberately reuse the existing directory.
**Reset setup** restores the form's initial values, including its run details
and initial preset or copied settings. After switching a preset or resetting,
**Undo** restores the previous draft, including any unapplied JSON text. Undo is
available until the next configuration edit or replacement; it is not a full
edit history. Edited stage settings are labeled **Custom settings**.

Paths entered in the form are resolved from the repository root. Data and
session-list files can be outside the repository. Dashboard run outputs belong
in named direct child directories under this repository's `cache/` directory so the viewer
can find them (for example, `cache/next_dashboard_001`, not a nested run path). The dashboard does not upload recordings or move existing runs.
The optional session list filters the session files available in the selected
data directory; missing listed sessions are reported by the pipeline.

### Reuse a run directory

A nonempty run directory is rejected unless reuse is explicitly enabled in the
form. Enable reuse when resuming decoding or running later stages on existing
outputs. This permits stages to replace their own artifacts; it does not create
a second independent result snapshot.

Each pipeline invocation still receives its own manifest in `manifests/`.
Earlier invocation records survive partial reruns, while
`pipeline_manifest.json` points to the latest invocation. History describes what
ran and with which command and settings; it does not version every cache or
figure. Use a new run directory to preserve both result sets for comparison.
See [run manifest history](outputs.md#run-manifest-history).

## Follow processing

The processing view shows the requested stages, their actual manifest status,
finished-stage durations, and a live log. Stage completion is a count of finished
stages, not an estimate of total runtime: decoding and mixed-effects fits can
take much longer than other stages. The dashboard does not invent a percentage
for an unfinished stage.

One dashboard job runs at a time. Click **Stop run** to open its confirmation,
then **Yes, stop run** to interrupt its process group, including pipeline workers.
Choose **Keep running**, or press Escape while in the confirmation, to dismiss
it. Completed outputs and decoding checkpoints remain on disk. Review the log and
follow [resume guidance](configuration.md#resume-and-rerun) before reusing them.

Live updates use WebSocket messages. The visible log is bounded to its latest
500 lines; the complete combined stdout/stderr stream is saved at
`cache/.dashboard/<job-id>.log`. Job metadata is saved next to that log. The
exact stage settings submitted for each job are retained in
`configs/next/.dashboard/<job-id>.json`, and the pipeline manifest records the
Python command, working directory, and resolved settings. These generated
files are ignored by Git.

Use **Filter log lines** to find messages, **Wrap lines** to read long entries,
and **Copy log** or **Save log** to export the currently displayed buffered
lines. With a filter active, those exports contain only matching lines; they do
not retrieve the complete log file. **Follow output** keeps the latest output
visible. Scrolling upward pauses following so you can read earlier messages;
scroll to the bottom or enable it again to resume. Filtering temporarily pauses
following. If the live connection is interrupted, the view checks the API every
five seconds while reconnecting.

After a service restart, previous job records remain available. An unfinished
job is marked failed with an explanation; restarting the dashboard does not
automatically restart analysis. A browser refresh does not cancel a running job.
Do not start a separate command-line writer against the same run directory while
a dashboard job is using it.

## Find existing runs

You do not need to import or register runs. The **Run library** scans named
directories directly inside this repository's `cache/`, including analyses
started from the command line. For example:

```text
cache/
  next_dashboard_001/
    pipeline_manifest.json
    select/
    decode/
    states/
```

A folder is discoverable when it contains at least one recognized stage
directory, or a readable `pipeline_manifest.json` with nonempty `run_id` and
`runner_config` fields. This allows a new run to appear before its first stage
finishes. Partial and failed runs can appear too; a listing does not certify
that every output is complete or compatible.

The scan skips hidden folders and symlinks. It does not descend into nested
run folders such as `cache/project/run1/`, and no particular run-name prefix is
required. Historical flat caches are not migrated by the viewer. Dashboard
job records supply friendly display names when available; other runs use their
folder names.

Open the run library or click its **Refresh runs** icon after adding a run or
finishing a command-line analysis. The library also loads when the dashboard
first opens. Refresh also reloads the selected run and session results. Search
by run name or cache path, filter by latest invocation status, and sort by
**Newest first** or **Name A–Z**. **Reset filters** shows all runs again. Filtering
the library keeps an already selected run open below it. Select a run to inspect
sessions, invocation history, and artifacts.

## Inspect and compare results

The session viewer shows decoding confidence over time, observed evaluation
metrics when available, and off-state durations. Missing or incompatible data
are reported explicitly. Evaluation and state summaries are hidden when their
saved cue, trial rows, or time bins disagree with the decoding cache; state
fingerprints are checked when present. The viewer does not replace the
pipeline's full provenance validation or recompute scientific results.

The confidence chart averages saved probabilities across the decoded trials at
each bin start. Its null interval is the 2.5th–97.5th percentile across
shuffle-specific trial means. This descriptive band is **not** the corrected
state-detection threshold or a confidence interval for the observed curve. See
[decoding methods](methods.md#decode) and [state methods](methods.md#states).

Switch between **Overview**, **Figures**, **Tables**, and **Run history** to
explore a selected run. When a tab has keyboard focus, use the arrow keys to
move between tabs, or Home and End to jump to the first or last tab. Session
selectors apply to Overview and Figures. Tables and Run history browse outputs
and invocation records for the whole run.

Use the comparison workspace to place two session/run selections side by side:

- Choose **Across sessions** within one run to inspect differences in cue
  populations, confidence, performance, and state durations.
- Choose **Across runs** and use **Match session A** to select the same recording
  on the right when it is available. Matching prefers the same cue; if only a
  different cue is available, the comparison warns about that difference. You
  can still select both sessions independently.
- Compare saved figures through their stage and session filters, and inspect
  CSV results in the table viewer or download the original artifact.

Use **Split screen** for separate session views or **Overlay curves** to compare their
confidence curves on one chart. The swap button exchanges the two sides.

Comparison warnings identify different preferred cues, different trial sets for
the same session, and different time grids. Trials are compared by their saved
IDs; a change in ordering alone does not mean a different set. Each curve keeps
its own recorded time coordinates. The dashboard does not pair trials, align
populations, pool sessions, or perform a new statistical test between runs.

Across-run comparisons also list configuration differences using the latest
recorded settings for each stage across manifest history. Cache and data paths
are omitted. Missing history is shown as unrecorded; an empty difference list
does not establish that the underlying data or output provenance are identical.

The **Figure formats** controls offer PNG, TIFF, EPS, and PDF for every plotting
stage. Choose **PNG and PDF** for inline previews plus PDF originals. PDF keeps
vector paths and text with transparency and lossless compression; image-based
plots still contain raster elements. See [figure exports](configuration.md#figure-exports).

PNG figures display directly. PDF-only runs are supported and their figures
remain available through **Download original**, without conversion. TIFF and
EPS also use downloads for an external viewer. Existing output files in formats
not selected for a rerun remain on disk.

Open a previewable figure to inspect it, then drag the **Zoom** slider or use its
arrow keys. The percentage shows the current zoom; **Reset zoom** returns it to
100%. Zoom changes only the preview, and **Download original** retains the saved
figure unchanged.

CSV tables can be paged through and downloaded from the Tables view. Its
**Supporting files** section downloads saved JSON and log artifacts within the
run directory. Manifest records are also readable in Run history. Dashboard
processing logs remain separately in `cache/.dashboard/`. The result API does
not serve pickle files for download.

## Generated files and reusable presets

The dashboard keeps its generated files locally. They remain on disk when
ignored by Git; ignoring them does not affect analysis or run discovery.

| Location | What it contains | How to treat it |
| --- | --- | --- |
| `configs/next/.dashboard/<job-id>.json` | The exact stage settings submitted for one job | Retain with a run to replay its recorded command verbatim |
| `cache/.dashboard/` | Job records, friendly names, and full processing logs | Keep for dashboard job history; not required to discover stage-layout runs |
| `cache/<run-name>/` | Scientific outputs and per-invocation manifests | Archive the whole run directory to preserve results and their history |
| `dashboard/node_modules/` | Installed frontend dependencies | Recreate with `npm --prefix dashboard ci` |
| `dashboard/dist/` and `dashboard/*.tsbuildinfo` | Built frontend and TypeScript build bookkeeping | Recreate with `npm --prefix dashboard run build` after installing dependencies |
| `site/` | Built MkDocs guide, including its search index | Recreate with `uv run --group dashboard --group docs --locked mkdocs build --strict` |

To keep a reusable preset in Git, save a named JSON such as
`configs/next/my_analysis.json` outside `.dashboard/`. The supplied example and
smoke presets are maintained this way. Run manifests retain resolved settings
and the exact invocation, but the generated `--settings` file is also needed to
repeat that command unchanged. Copying a run directory alone does not copy its
dashboard job records or generated settings file.

The ignore rules also reserve `dashboard/playwright-report/` and
`dashboard/test-results/` for generated browser-test output. They are
precautionary; there is currently no Playwright setup. Conversely,
`dashboard/src/lib/` is explicitly **included** in Git because it contains
frontend source, despite the repository's older generic `lib/` ignore rule.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `npm: command not found` or an unsupported Node version | Select Node.js 24 with nvm as shown above, then retry setup. If nvm is already installed but unavailable in this terminal, load it as described below. |
| Python cannot find `scripts.next.dashboard` | Run the start command from the repository root, not from `dashboard/` or `scripts/next/`. |
| Missing `fastapi`, `uvicorn`, or `mkdocs` | Use the documented command with both `--group dashboard --group docs`; the docs group is required for building the guide. |
| Browser says it cannot connect | Check that the server terminal is still running and that the browser port matches `--port`. Inspect the terminal for a startup error. |
| `Frontend is not built` or `Documentation is not built` | Follow [first-time setup](#first-time-setup) with `--build`. The missing guide page also gives a docs-only build command. |
| `Address already in use` | Use a free dashboard port. Both dashboard and guide share that port. See [Use another port](#use-another-port). |
| Recent frontend changes are missing | Rebuild with `npm --prefix dashboard run build`, then refresh the browser. Use [frontend development](#frontend-development) for automatic updates while editing. |
| Guide content is old, or a new Methods link gives 404 | Rebuild the guide with `uv run --group dashboard --group docs --locked mkdocs build --strict`, then refresh it. A missing documentation page never opens the dashboard in its place. |
| A saved run is missing from the library | Click **Refresh runs**, then check its location and layout against [Find existing runs](#find-existing-runs). |
| A discovered run reports missing or incompatible results | Read the displayed error and follow the [cache and provenance troubleshooting](troubleshooting.md#a-cache-is-incompatible); discovery does not validate every stage. |

For an existing nvm installation at its usual location, load it into the
current terminal before selecting Node:

```bash
source "$HOME/.nvm/nvm.sh"
nvm use 24
```

If Node.js 24 has not been installed through nvm, run `nvm install 24` first.
Once the Python server is running, its
[health endpoint](http://127.0.0.1:8000/api/health) should show `"status": "ok"`.
Use your chosen port if you changed it. A healthy backend with a missing
interface usually means the frontend build step is still needed.

## Frontend development

Use this workflow when changing the React interface. Normal dashboard use only
needs the single Python server described above.

In terminal 1, from the repository root, start the backend on port 8000:

```bash
uv run --group dashboard --group docs --locked python -m scripts.next.dashboard
```

In terminal 2, also from the repository root, start Vite:

```bash
npm --prefix dashboard ci
npm --prefix dashboard run dev
```

You can skip `npm ci` when dependencies are already installed and unchanged.
Open **[http://127.0.0.1:5173/](http://127.0.0.1:5173/)** for this workflow.
Keep both terminals running. Vite reloads frontend edits and forwards REST,
WebSocket, and `/docs/` requests to the backend on port 8000. It does not replace
the Python backend. If you change that backend port, update the proxy targets in
`dashboard/vite.config.ts` to match.

The frontend uses React, TypeScript, Vite, Tailwind CSS, and shadcn-style Radix
components. The shared dropdown wraps
[Radix Select](https://www.radix-ui.com/primitives/docs/components/select), which
provides keyboard navigation, typeahead, and focus handling; keep these behaviors
when changing its appearance. The backend lives in `scripts/next/dashboard/`
and uses FastAPI and Pydantic. The pipeline guide is at `/docs/`; the developer
API reference is at `/api/docs`, with OpenAPI JSON at `/api/openapi.json` and
ReDoc at `/api/redoc`.
Tests and a
production build can be run with:

```bash
uv run --group dashboard --group docs --locked python -m unittest discover -s tests/next -v
npm --prefix dashboard test
npm --prefix dashboard run build
```

To refresh the integrated guide during development, run
`uv run --group dashboard --group docs --locked mkdocs build --strict` and
reload its tab; the backend does not need to restart. For automatic Markdown
reload, use the separate [MkDocs preview](../development.md#serve-the-documentation)
on port 8001. To make Vite's help links open that preview, start Vite with:

```bash
VITE_DOCS_BASE_URL=http://127.0.0.1:8001/ npm --prefix dashboard run dev
```

`VITE_DOCS_BASE_URL` is a frontend build/development setting, not a runtime Python
option. Omit it for the integrated `/docs/` guide. It also accepts an HTTPS
project URL for a separately hosted guide; include the repository subpath and
use documentation matching the local pipeline version.
