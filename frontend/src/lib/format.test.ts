import { describe, expect, it } from "vitest";

import { makePassage } from "@/test/fixtures";

import type { LibraryDocument } from "@/api/types";

import {
  docTypeLabel,
  documentBadge,
  documentDetails,
  retrievalLine,
  sourceLocation,
  sourceTitle,
  uploadLimits,
} from "./format";
import { buildScopes } from "./scopes";

function makeDocument(overrides: Partial<LibraryDocument> = {}): LibraryDocument {
  return {
    doc_id: "doc",
    title: "notes.md",
    collection_id: "uploads",
    doc_type: "runbook",
    doc_date: null,
    metadata: {},
    filename: "notes.md",
    ingested_at: "2025-11-01T09:30:00Z",
    chunks: 3,
    ...overrides,
  };
}

describe("retrievalLine", () => {
  it("leads with the re-ranker rank, which sets the order", () => {
    expect(retrievalLine(makePassage(2))).toBe("rerank #2 · bm25 – · dense #2 · rrf 0.0323");
  });

  it("leaves the re-ranker out when it is off", () => {
    const p = makePassage(2);
    const off = { ...p, retrieval: { ...p.retrieval, rerank_rank: null } };

    expect(retrievalLine(off)).toBe("bm25 – · dense #2 · rrf 0.0323");
  });
});

describe("uploadLimits", () => {
  it("names each format once and states the size limit", () => {
    const config = { upload_types: [".htm", ".html", ".md", ".txt"], max_upload_mb: 20 };

    expect(uploadLimits(config)).toBe("HTML, Markdown or TXT · up to 20 MB each");
  });

  it("falls back to the suffix for a format it has no name for", () => {
    expect(uploadLimits({ upload_types: [".csv"], max_upload_mb: 5 })).toBe(
      "CSV · up to 5 MB each",
    );
  });
});

describe("sourceLocation", () => {
  it("joins the heading path and the page", () => {
    const passage = { ...makePassage(1), section_path: ["Install", "Linux"], page: 4 };

    expect(sourceLocation(passage)).toBe("Install › Linux · Page 4");
  });

  it("is empty for a passage under no heading", () => {
    expect(sourceLocation({ ...makePassage(1), section_path: [] })).toBe("");
  });
});

describe("docTypeLabel", () => {
  it("names SEC forms and leaves other types as entered", () => {
    expect(docTypeLabel("10-K")).toBe("Form 10-K");
    expect(docTypeLabel("runbook")).toBe("runbook");
  });
});

describe("sourceTitle", () => {
  it("shows a filing by ticker, form and fiscal year", () => {
    expect(sourceTitle(makePassage(1))).toBe("ACME · Form 10-K · FY2025");
  });

  it("shows any other document by title and date", () => {
    const passage = { ...makePassage(1), title: "Expense policy", metadata: {} };

    expect(sourceTitle(passage)).toBe("Expense policy · 2025-09-27");
  });
});

describe("library labels", () => {
  it("badges a filing with its ticker and others with their file type", () => {
    expect(documentBadge(makeDocument({ metadata: { ticker: "ACME" } }))).toBe("ACME");
    expect(documentBadge(makeDocument())).toBe("MD");
  });

  it("dates a document by its own date, or by when it was uploaded", () => {
    const now = new Date("2025-11-20T12:00:00Z");

    expect(documentDetails(makeDocument({ doc_date: "2024-03-01" }), now)).toBe(
      "2024-03-01 · 3 chunks",
    );
    expect(documentDetails(makeDocument(), now)).toBe("Uploaded Nov 1 · 3 chunks");
  });
});

describe("buildScopes", () => {
  it("offers every collection in the library once, sorted", () => {
    const documents = [
      makeDocument({ collection_id: "sec-10k" }),
      makeDocument({ collection_id: "books" }),
      makeDocument({ collection_id: "sec-10k" }),
    ];

    expect(buildScopes(documents).map((scope) => [scope.label, scope.filters])).toEqual([
      ["All documents", undefined],
      ["books", { collections: ["books"] }],
      ["sec-10k", { collections: ["sec-10k"] }],
    ]);
  });
});
