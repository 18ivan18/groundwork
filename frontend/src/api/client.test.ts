import { afterEach, describe, expect, it, vi } from "vitest";

import { makeResponse } from "@/test/fixtures";

import { ApiError, askQuestion, getConfig, getDocuments, uploadDocument } from "./client";
import type { LibraryDocument } from "./types";

const DOCUMENT: LibraryDocument = {
  doc_id: "doc",
  title: "ACME Corp · Form 10-K",
  collection_id: "sec-10k",
  doc_type: "10-K",
  doc_date: "2025-09-27",
  metadata: { company: "ACME Corp", ticker: "ACME" },
  filename: "acme.txt",
  ingested_at: "2025-11-01T09:30:00Z",
  chunks: 12,
};

function stubFetch(response: Response | Error) {
  const mock = vi.fn(() =>
    response instanceof Error ? Promise.reject(response) : Promise.resolve(response),
  );
  vi.stubGlobal("fetch", mock);
  return mock;
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

async function failure(call: () => Promise<unknown>): Promise<ApiError> {
  const error: unknown = await call().catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(ApiError);
  return error as ApiError;
}

describe("api client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns the parsed body", async () => {
    stubFetch(json([DOCUMENT]));

    expect(await getDocuments()).toEqual([DOCUMENT]);
  });

  it("reads the upload limits from the config endpoint", async () => {
    const mock = stubFetch(json({ upload_types: [".md", ".txt"], max_upload_mb: 20 }));

    expect(await getConfig()).toEqual({ upload_types: [".md", ".txt"], max_upload_mb: 20 });
    const [path] = mock.mock.calls[0] as unknown as [string];
    expect(path).toBe("/api/config");
  });

  it("sends the current user on every request, alongside the caller's headers", async () => {
    const mock = stubFetch(json(makeResponse()));
    localStorage.setItem("groundwork.user", "hana");

    await askQuestion({ question: "Who sells anvils?" });

    const [, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    const headers = new Headers(init.headers);
    expect(headers.get("X-User")).toBe("hana");
    expect(headers.get("Content-Type")).toBe("application/json");
    localStorage.clear();
  });

  it("acts as alice until a user is chosen", async () => {
    const mock = stubFetch(json([]));

    await getDocuments();

    const [, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    expect(new Headers(init.headers).get("X-User")).toBe("alice");
  });

  it("does not warn about a response that matches its schema", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    stubFetch(json(makeResponse()));

    await askQuestion({ question: "Who sells anvils?" });

    expect(warn).not.toHaveBeenCalled();
  });

  it("warns about an unexpected response but still returns it", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const renamed = { ...DOCUMENT, chunks: undefined, chunk_count: 12 };
    stubFetch(json([renamed]));

    expect(await getDocuments()).toEqual([renamed]);
    expect(warn).toHaveBeenCalledOnce();
    expect(warn.mock.calls[0]?.[0]).toBe("Unexpected response from /api/documents");
  });

  it("posts questions as JSON", async () => {
    const mock = stubFetch(json(makeResponse()));

    await askQuestion({ question: "Who sells anvils?" });

    const [path, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    expect(path).toBe("/api/ask");
    expect(init.body).toBe('{"question":"Who sells anvils?"}');
  });

  it("surfaces the friendly detail from the API", async () => {
    stubFetch(json({ detail: "The LLM provider rejected the API key." }, 502));

    const error = await failure(getDocuments);

    expect(error.status).toBe(502);
    expect(error.message).toBe("The LLM provider rejected the API key.");
  });

  it("hides FastAPI's validation list", async () => {
    stubFetch(json({ detail: [{ loc: ["body"], msg: "too short", type: "string" }] }, 422));

    expect((await failure(getDocuments)).message).toBe(
      "Please check what you entered and try again.",
    );
  });

  it("copes with an HTML error page from a proxy", async () => {
    stubFetch(new Response("<html>Bad gateway</html>", { status: 502 }));

    expect((await failure(getDocuments)).message).toContain("status 502");
  });

  it("reports a network failure without the raw error", async () => {
    stubFetch(new TypeError("Failed to fetch"));

    const error = await failure(getDocuments);

    expect(error.status).toBe(0);
    expect(error.message).toBe("Could not reach the server. Check that the API is running.");
  });

  it("reports a success body that is not JSON", async () => {
    stubFetch(new Response("<html></html>", { status: 200 }));

    expect((await failure(getDocuments)).message).toContain("could not be read");
  });

  it("leaves an empty date out of an upload", async () => {
    const mock = stubFetch(json({ job_id: "j", doc_id: "d" }, 202));
    const file = new File(["text"], "notes.txt");

    await uploadDocument(file, { collection: "uploads", doc_type: "notes", doc_date: "" });

    const [, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    const form = init.body as FormData;
    expect(form.get("collection")).toBe("uploads");
    expect(form.get("doc_type")).toBe("notes");
    expect(form.has("doc_date")).toBe(false);
  });
});
