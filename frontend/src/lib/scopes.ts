import type { AskFilters, LibraryDocument } from "@/api/types";

export interface Scope {
  value: string;
  label: string;
  filters?: AskFilters;
}

export const ALL_DOCUMENTS: Scope = { value: "all", label: "All documents" };

/** Build the "Search in" options from the collections really in the library. */
export function buildScopes(documents: LibraryDocument[]): Scope[] {
  const collections = [...new Set(documents.map((document) => document.collection_id))].sort();
  return [
    ALL_DOCUMENTS,
    ...collections.map((collection) => ({
      value: collection,
      label: collection,
      filters: { collections: [collection] },
    })),
  ];
}
