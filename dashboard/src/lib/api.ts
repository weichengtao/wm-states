export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch("/api" + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON errors */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
export const runPath = (id: string) => `/runs/${encodeURIComponent(id)}`;
export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
