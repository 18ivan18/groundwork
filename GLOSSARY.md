# Groundwork

A retrieval-augmented assistant that answers questions over document collections, with per-user access
control, and measures its own retrieval and answer quality.

## Data

**Collection**:
A named group of documents that is indexed and searched together, such as `engineering` or `northwind`.
_Avoid_: dataset, folder

**Corpus**:
The documents of one collection that the demo indexes and the eval runs against, described by its manifest.
_Avoid_: test data, raw data

**Manifest**:
The record of a corpus: its source, licence, revision, and one entry per document with its key, access
groups and content hash.
_Avoid_: index, catalogue

**Corpora version**:
A numbered, unchanging release of all corpora together; every eval result names the one it was measured on.
_Avoid_: dataset version, data release, snapshot

**Corpus sync**:
The step that makes an index hold exactly the corpus documents the manifests describe, and leaves uploads alone.
_Avoid_: re-ingest, refresh, import

**Upload**:
A document a user added. It has no manifest key, whichever collection it is in, so no corpus sync changes it.
_Avoid_: user document, custom document

**Publication**:
A state of an index's contents that keyword search is built from, named by its generation and a random token;
a write isn't searchable by keyword until it is published.
_Avoid_: release, bump, version

**Eval set**:
The questions written to measure retrieval and answers, each pointing at corpus documents by manifest key.
_Avoid_: test data, benchmark, golden set

**Test fixture**:
A small file that a unit test reads, independent of any corpus.
_Avoid_: test data, sample corpus
