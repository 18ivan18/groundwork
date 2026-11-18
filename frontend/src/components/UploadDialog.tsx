import { useQuery } from "@tanstack/react-query";
import { FileUp, X } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";

import { getConfig } from "@/api/client";
import type { UploadTags } from "@/api/types";
import { UploadQueueItem } from "@/components/UploadQueueItem";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { UploadQueue } from "@/hooks/useUploadQueue";
import { plural, uploadLimits } from "@/lib/format";

const FIELD_LABEL = "text-[13px] font-normal text-muted-foreground";
const FIELD_INPUT = "h-11 rounded-lg border-input bg-card";
// The API's rule for collection ids: lowercase words joined by hyphens.
const COLLECTION_ID = /^[a-z0-9]+(-[a-z0-9]+)*$/;

/** Why the tags can't be uploaded yet, or null when they can. */
function tagProblem(tags: UploadTags): string | null {
  if (!COLLECTION_ID.test(tags.collection.trim())) {
    return "Name the collection in lowercase words joined by hyphens, such as team-notes.";
  }
  if (!tags.doc_type.trim()) return "Enter a document type to upload.";
  return null;
}

interface UploadDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  queue: UploadQueue;
}

export function UploadDialog({ open, onOpenChange, queue }: UploadDialogProps) {
  const [tags, setTags] = useState<UploadTags>({ collection: "uploads", doc_type: "", doc_date: "" });
  // The server's limits only change on a restart. Until they load, any file can be picked: the
  // upload endpoint checks the same limits itself.
  const config = useQuery({ queryKey: ["config"], queryFn: getConfig, staleTime: Infinity });
  // Radix only returns focus to a DialogTrigger, and this dialog opens from more than one place.
  const opener = useRef<HTMLElement | null>(null);
  const readyCount = queue.rows.filter((row) => row.view.kind === "ready").length;
  const problem = tagProblem(tags);

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    queue.addFiles(Array.from(event.dataTransfer.files));
  }

  function cancel() {
    queue.discardReady();
    onOpenChange(false);
  }

  function done() {
    queue.clearFinished();
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        aria-modal="true"
        showCloseButton={false}
        onOpenAutoFocus={() => {
          opener.current = document.activeElement as HTMLElement | null;
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          opener.current?.focus();
        }}
        className="max-h-[calc(100vh-2rem)] gap-5 overflow-y-auto rounded-[14px] bg-card p-7 sm:max-w-[640px]"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex flex-col gap-1">
            <DialogTitle className="text-[22px] font-semibold">
              Add documents to the library
            </DialogTitle>
            <DialogDescription className="text-sm leading-normal text-muted-foreground">
              Each file is split into passages, embedded and added to both the keyword and the
              vector index.
            </DialogDescription>
          </div>
          <DialogClose asChild>
            <Button variant="ghost" aria-label="Close" className="size-11 flex-none p-0">
              <X aria-hidden="true" />
            </Button>
          </DialogClose>
        </div>

        <div
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
          className="flex flex-col items-center gap-2.5 rounded-xl border-[1.5px] border-dashed border-dropzone-border bg-dropzone-bg px-5 py-7 text-center"
        >
          <FileUp aria-hidden="true" className="size-8 text-muted-foreground" strokeWidth={1.6} />
          <p className="text-base font-medium">Drag files here, or</p>
          <label
            htmlFor="file-input"
            className="inline-flex min-h-11 cursor-pointer items-center rounded-lg border border-primary px-[18px] text-sm font-semibold text-primary focus-within:ring-[3px] focus-within:ring-ring/50"
          >
            Choose files
            <input
              id="file-input"
              type="file"
              multiple
              accept={config.data?.upload_types.join(",")}
              onChange={(event) => {
                queue.addFiles(Array.from(event.target.files ?? []));
                // Lets the same file be chosen again after it is removed.
                event.target.value = "";
              }}
              className="sr-only"
            />
          </label>
          {config.data && (
            <p className="text-[13px] text-muted-foreground">{uploadLimits(config.data)}</p>
          )}
        </div>

        <fieldset className="grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-3">
          <legend className="pb-2.5 text-sm font-semibold">Tags for filtering and citations</legend>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="collection" className={FIELD_LABEL}>
              Collection
            </Label>
            <Input
              id="collection"
              required
              maxLength={40}
              value={tags.collection}
              onChange={(event) => setTags({ ...tags, collection: event.target.value })}
              className={FIELD_INPUT}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="doctype" className={FIELD_LABEL}>
              Document type
            </Label>
            <Input
              id="doctype"
              required
              maxLength={50}
              placeholder="runbook, policy, report…"
              value={tags.doc_type}
              onChange={(event) => setTags({ ...tags, doc_type: event.target.value })}
              className={FIELD_INPUT}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="docdate" className={FIELD_LABEL}>
              Date
            </Label>
            <Input
              id="docdate"
              type="date"
              value={tags.doc_date}
              onChange={(event) => setTags({ ...tags, doc_date: event.target.value })}
              className={FIELD_INPUT}
            />
          </div>
        </fieldset>

        {queue.rows.length > 0 && (
          <div className="flex flex-col gap-2">
            <h3 className="text-sm font-semibold">Queue</h3>
            <ul className="flex flex-col gap-2">
              {queue.rows.map((row) => (
                <UploadQueueItem key={row.item.id} row={row} onRemove={queue.remove} />
              ))}
            </ul>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <p className="text-xs text-muted-foreground">
            {readyCount > 0 && problem
              ? problem
              : "You can close this window. Indexing continues in the background."}
          </p>
          <div className="flex gap-2.5">
            <Button variant="outline" onClick={cancel}>
              Cancel
            </Button>
            {readyCount > 0 ? (
              <Button
                disabled={problem !== null}
                onClick={() =>
                  void queue.submit({
                    ...tags,
                    collection: tags.collection.trim(),
                    doc_type: tags.doc_type.trim(),
                  })
                }
                className="font-semibold"
              >
                Upload {plural(readyCount, "file")}
              </Button>
            ) : (
              <Button onClick={done} className="font-semibold">
                Done
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
