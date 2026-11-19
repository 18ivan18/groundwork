"""Search the active index and print the passages.

Run from backend/: uv run python scripts/search.py "your question" [--k 6] [--mode reranked]
"""

import argparse
import asyncio
from typing import get_args

from groundwork.config import Settings
from groundwork.container import bm25_root, build_reranker, make_retriever, open_app_store
from groundwork.domain.models import section_label
from groundwork.domain.users import access_for, load_users
from groundwork.retrieval.hybrid import SearchMode
from groundwork.retrieval.sparse import SparseCache


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--mode", choices=get_args(SearchMode), default="reranked")
    parser.add_argument(
        "--user", help="Search as this user from conf/users.yaml. Without it, nothing is hidden."
    )
    args = parser.parse_args()

    settings = Settings()
    access = access_for(load_users(settings.users_file), args.user)
    if args.user is None:
        print("(as no user: every document is searched)")
    async with open_app_store(settings) as opened:
        sparse = SparseCache(opened.store, bm25_root(settings))
        retriever = make_retriever(
            settings, opened.embedder, opened.store, sparse, build_reranker(settings)
        )
        passages = await retriever.search(
            args.query, args.k or settings.top_k, mode=args.mode, access=access
        )

    for rank, passage in enumerate(passages, start=1):
        excerpt = " ".join(passage.chunk.text.split())[:240]
        print(
            f"{rank}. rerank={passage.rerank_rank} rrf={passage.rrf_score:.4f} "
            f"dense={passage.dense_rank} "
            f"sparse={passage.sparse_rank}  {passage.document.title} | "
            f"{section_label(passage.chunk)}"
        )
        print(f"   {excerpt}")


if __name__ == "__main__":
    asyncio.run(main())
