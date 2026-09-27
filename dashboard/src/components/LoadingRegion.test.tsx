import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LoadingRegion, observeContentHeight } from "./LoadingRegion";
import { Empty, Notice } from "./shared";

afterEach(() => vi.unstubAllGlobals());

describe("loading regions", () => {
  it("shows the new request status without exposing stale scientific content", () => {
    const html = renderToStaticMarkup(
      <LoadingRegion loading label="Reading session B…">
        <p>Session A confidence: 0.95</p>
      </LoadingRegion>,
    );
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('role="status"');
    expect(html).toContain("Reading session B…");
    expect(html).not.toContain("Session A");
    expect(html).not.toContain("min-height");
  });

  it("allows settled empty/error results to replace the loading state naturally", () => {
    const html = renderToStaticMarkup(
      <LoadingRegion loading={false} className="session-results">
        <Notice>Could not read session B</Notice>
        <Empty title="No session results" />
      </LoadingRegion>,
    );
    expect(html).toContain('class="loading-region session-results"');
    expect(html).toContain('aria-busy="false"');
    expect(html).toContain('role="alert"');
    expect(html).toContain("No session results");
    expect(html).not.toContain("min-height");
    expect(html).not.toContain("Loading results");
  });

  it("can reserve an outer layout while nested regions show only their own loading state", () => {
    const html = renderToStaticMarkup(
      <LoadingRegion loading retainChildren>
        <LoadingRegion loading={false}>
          <p>Session A confidence: 0.95</p>
        </LoadingRegion>
        <LoadingRegion loading label="Reading session B…">
          <p>Stale session C confidence: 0.1</p>
        </LoadingRegion>
      </LoadingRegion>,
    );
    expect(html.match(/aria-busy="true"/g)).toHaveLength(2);
    expect(html).toContain("Session A confidence: 0.95");
    expect(html).toContain("Reading session B…");
    expect(html).not.toContain("Stale session C");
    expect(html).not.toContain("Loading results");
    expect(html.match(/role="status"/g)).toHaveLength(1);
  });

  it("tracks both growing and shrinking content and ignores queued callbacks after cleanup", () => {
    let height = 600.2;
    let notifyResize = () => {};
    const disconnect = vi.fn();
    const observe = vi.fn();
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(callback: () => void) {
          notifyResize = callback;
        }
        observe = observe;
        disconnect = disconnect;
      },
    );
    const view = { addEventListener: vi.fn(), removeEventListener: vi.fn() };
    const element = {
      getBoundingClientRect: () => ({ height }),
      ownerDocument: { defaultView: view },
    } as unknown as HTMLElement;
    const onHeight = vi.fn();
    const stop = observeContentHeight(element, onHeight);
    expect(observe).toHaveBeenCalledWith(element);
    expect(onHeight).toHaveBeenLastCalledWith(601);
    height = 720;
    notifyResize();
    expect(onHeight).toHaveBeenLastCalledWith(720);
    height = 250;
    notifyResize();
    expect(onHeight).toHaveBeenLastCalledWith(250);
    expect(view.addEventListener).not.toHaveBeenCalled();
    stop();
    height = 900;
    notifyResize();
    expect(disconnect).toHaveBeenCalledOnce();
    expect(onHeight).toHaveBeenCalledTimes(3);
  });

  it("keeps a resize fallback and cleans it up without ResizeObserver", () => {
    vi.stubGlobal("ResizeObserver", undefined);
    let height = 500;
    const view = { addEventListener: vi.fn(), removeEventListener: vi.fn() };
    const element = {
      getBoundingClientRect: () => ({ height }),
      ownerDocument: { defaultView: view },
    } as unknown as HTMLElement;
    const onHeight = vi.fn();
    const stop = observeContentHeight(element, onHeight);
    expect(view.addEventListener).toHaveBeenCalledWith(
      "resize",
      expect.any(Function),
    );
    const resize = view.addEventListener.mock.calls[0][1] as () => void;
    height = 650;
    resize();
    expect(onHeight).toHaveBeenLastCalledWith(650);
    stop();
    expect(view.removeEventListener).toHaveBeenCalledWith("resize", resize);
  });
});
