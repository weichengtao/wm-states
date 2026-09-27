import { afterEach, describe, expect, it, vi } from "vitest";
import { copyText } from "./clipboard";

afterEach(() => vi.unstubAllGlobals());

describe("copying on local, HTTPS, and tailnet HTTP pages", () => {
  it("uses the available clipboard and waits for success", async () => {
    let finish!: () => void;
    const writeText = vi.fn(
      () => new Promise<void>((resolve) => (finish = resolve)),
    );
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    let finished = false;
    const copying = copyText("python -m scripts.next.pipeline\n").then(
      (copied) => {
        finished = true;
        return copied;
      },
    );
    expect(writeText).toHaveBeenCalledWith("python -m scripts.next.pipeline\n");
    expect(finished).toBe(false);
    finish();
    expect(await copying).toBe(true);
  });

  it("requests manual copy when a remote HTTP page has no Clipboard API", async () => {
    vi.stubGlobal("navigator", {});
    expect(await copyText("A command")).toBe(false);
    vi.stubGlobal("navigator", undefined);
    expect(await copyText("A command")).toBe(false);
  });

  it("requests manual copy when browser permission is denied", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("NotAllowedError"));
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    expect(await copyText("A command")).toBe(false);
  });

  it("handles synchronous clipboard failures and blocked property access", async () => {
    vi.stubGlobal("navigator", {
      clipboard: {
        writeText() {
          throw new Error("NotAllowedError");
        },
      },
    });
    expect(await copyText("A command")).toBe(false);
    vi.stubGlobal("navigator", {
      get clipboard() {
        throw new Error("SecurityError");
      },
    });
    expect(await copyText("A command")).toBe(false);
  });
});
