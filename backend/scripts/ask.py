"""Ask a question end to end. This calls the configured LLM provider, which may be paid.

Run from backend/: uv run python scripts/ask.py "your question"
"""

import argparse
import asyncio
import sys

from groundwork.config import Settings
from groundwork.container import build_services, open_app_store
from groundwork.domain.users import access_for, load_users
from groundwork.llm.base import LLMError
from groundwork.qa.prompt import passage_source


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument(
        "--user", help="Search as this user from conf/users.yaml. Without it, nothing is hidden."
    )
    args = parser.parse_args()

    settings = Settings()
    access = access_for(load_users(settings.users_file), args.user)
    if args.user is None:
        print("(as no user: every document is searched)")
    async with open_app_store(settings) as opened:
        service = build_services(settings, opened.store, opened.embedder).qa
        try:
            answer = await service.ask(args.question, access=access)
        except LLMError as error:
            sys.exit(f"FAILED: {error}")

    print(answer.text if not answer.insufficient_evidence else "[not enough evidence]")
    for source in answer.sources:
        print(f"  [{source.id}] {passage_source(source.passage)}")
    print(
        f"  ({len(answer.sources)} cited of {len(answer.retrieved)} retrieved, "
        f"retrieval {answer.retrieval_ms} ms, generation {answer.generation_ms} ms)"
    )


if __name__ == "__main__":
    asyncio.run(main())
