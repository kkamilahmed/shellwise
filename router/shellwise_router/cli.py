"""Route one request from the command line: ``shellwise-route "is anything on port 3000"``."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .router import ToolRouter


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("request")
    ap.add_argument(
        "--catalog", type=Path, default=Path(__file__).resolve().parent.parent / "catalog.json"
    )
    ap.add_argument("--device")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = ToolRouter(args.catalog, device=args.device).route(args.request)
    if args.json:
        print(json.dumps(asdict(r), indent=1))
    else:
        best = max(r.tool_probs, key=r.tool_probs.get) if r.tool_probs else "?"
        tool = r.tool or f"(unsure, best guess {best})"
        print(f"bucket: {r.bucket} ({r.bucket_confidence})")
        print(f"tool:   {tool} ({r.tool_confidence})  {r.ms} ms")


if __name__ == "__main__":
    main()
