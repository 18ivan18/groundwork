// Response types are inferred from the runtime schemas in schemas.ts. Request types are written
// here: the client builds them, so there is nothing to check at runtime.
import type { z } from "zod";

import type {
  AskResponseSchema,
  ClientConfigSchema,
  HealthSchema,
  JobSchema,
  JobStatusSchema,
  LibraryDocumentSchema,
  PassageSchema,
  SourceSchema,
  UploadResponseSchema,
} from "./schemas";

export type JobStatus = z.infer<typeof JobStatusSchema>;

export type Health = z.infer<typeof HealthSchema>;
export type ClientConfig = z.infer<typeof ClientConfigSchema>;
export type LibraryDocument = z.infer<typeof LibraryDocumentSchema>;
export type Passage = z.infer<typeof PassageSchema>;
export type Source = z.infer<typeof SourceSchema>;
export type AskResponse = z.infer<typeof AskResponseSchema>;
export type UploadResponse = z.infer<typeof UploadResponseSchema>;
export type Job = z.infer<typeof JobSchema>;

export interface AskFilters {
  collections?: string[];
  doc_types?: string[];
  /** Inclusive ISO dates. Undated documents never match a range. */
  date_from?: string;
  date_to?: string;
  /** Every key must have exactly this value in the document's metadata. */
  metadata?: Record<string, string>;
}

export interface AskRequest {
  question: string;
  filters?: AskFilters;
  top_k?: number;
}

export interface UploadTags {
  collection: string;
  doc_type: string;
  /** An ISO date, or "" for none. */
  doc_date: string;
}
