#!/usr/bin/env python3
"""List the models your OPENAI_API_KEY can actually reach.

Model availability varies by account, so pick one from here and set it:

    export OPENAI_MODEL_ID=<id>
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY is not set.")
        return 1

    from openai import OpenAI

    ids = sorted(m.id for m in OpenAI(api_key=key).models.list())
    chat = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4"))]

    print(f"{len(ids)} models available; {len(chat)} look like chat models:\n")
    for model_id in chat:
        print(" ", model_id)
    print("\nPick one that supports tool calling, then:")
    print("  export OPENAI_MODEL_ID=<id>")
    print("  python chat.py --provider openai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
