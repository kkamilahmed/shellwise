"""Two-stage routing: bucket first, then a tool inside that bucket.

Stage one asks one Choice question over the routable buckets. Stage two asks
a Choice question over the bucket's common tools, using Laya's tournament
mode when there are more options than fit one question well. Both stages
report calibrated confidence; callers gate on it and fall back to free
generation when the router is unsure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .buckets import ROUTABLE

OTHER = "other"
DEFAULT_MODEL = "convaiinnovations/laya"
FLAT_LIMIT = 16  # above this many options, use a tournament

# Representative tools per bucket. Laya reads these as the option text for stage
# one; measured zero-shot they beat prose descriptions (0.53 vs 0.47 bucket accuracy).
BUCKET_HINTS: dict[str, str] = {
    "files": "ls cp mv rm mkdir find ln chmod chown du df rsync stat",
    "text": "cat grep sed awk sort uniq cut wc head tail diff tr less",
    "archive": "tar zip unzip gzip bzip2 xz",
    "process": "ps top kill pkill lsof nice nohup caffeinate",
    "network": "curl ssh scp ping ifconfig netstat dig nc traceroute",
    "system": "uname sw_vers date uptime pmset launchctl crontab sysctl log reboot",
    "disk": "diskutil mount umount hdiutil dd fsck tmutil",
    "users": "whoami id sudo su passwd who last security openssl",
    "dev": "git make gcc clang python3 swift xcodebuild docker",
    "shell": "cd echo export alias history which man xargs sleep seq read test",
    "macos": "open defaults osascript pbcopy pbpaste say screencapture mdfind",
    "media": "sips afplay textutil lp lpr lpstat",
    "data": "sqlite3 plutil base64 shasum md5 hexdump xxd bc uuidgen",
}


def bucket_criteria() -> dict[str, str]:
    return {b.id: BUCKET_HINTS[b.id] for b in ROUTABLE}


def tool_criteria(tools: dict[str, dict], bucket_id: str) -> dict[str, str]:
    crit = {
        name: (e.get("description") or name)[:60]
        for name, e in sorted(tools.items())
        if e["bucket"] == bucket_id and e.get("common")
    }
    crit[OTHER] = "none of these tools fits the request"
    return crit


@dataclass
class RouteResult:
    bucket: str
    bucket_confidence: float
    tool: str | None
    tool_confidence: float
    candidates: int
    ms: float
    bucket_probs: dict[str, float] = field(default_factory=dict)
    tool_probs: dict[str, float] = field(default_factory=dict)


class ToolRouter:
    def __init__(
        self,
        catalog: Path | dict,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        extra_tools: dict[str, dict] | None = None,
        bucket_threshold: float = 0.6,
        tool_threshold: float = 0.6,
    ):
        import laya

        data = json.loads(Path(catalog).read_text()) if not isinstance(catalog, dict) else catalog
        self.tools: dict[str, dict] = dict(data["tools"])
        for name, entry in (extra_tools or {}).items():  # plugin registrations
            self.tools[name] = {**entry, "common": True}
        self.bucket_threshold = bucket_threshold
        self.tool_threshold = tool_threshold
        self.agent = laya.load(model, device=device) if device else laya.load(model)
        self._laya = laya
        self._bucket_q = {
            "bucket": {
                "type": "choice",
                "instructions": "Which area does this shell request belong to?",
                "criteria": bucket_criteria(),
            }
        }
        self._tool_q: dict[str, dict] = {}

    def _tool_question(self, bucket_id: str) -> dict:
        if bucket_id not in self._tool_q:
            self._tool_q[bucket_id] = {
                "tool": {
                    "type": "choice",
                    "instructions": "Which command-line tool does this request need?",
                    "criteria": tool_criteria(self.tools, bucket_id),
                }
            }
        return self._tool_q[bucket_id]

    def _ask(self, state: dict, question: dict) -> dict:
        n = len(next(iter(question.values()))["criteria"])
        if n > FLAT_LIMIT:
            res = self._laya.predict_tournament(self.agent, state, question)
        else:
            res = self.agent.predict(state, question)
        return next(iter(res["answers"].values()))

    def route(self, request: str, context: str | None = None) -> RouteResult:
        t0 = time.perf_counter()
        state = {"request": request}
        if context:
            state["context"] = context
        b = self._ask(state, self._bucket_q)
        bucket_id = b["choice"]
        b_conf = float(b.get("answer_confidence", b["confidence"]))
        tool, t_conf, t_probs, n = None, 0.0, {}, 0
        if b_conf >= self.bucket_threshold:
            q = self._tool_question(bucket_id)
            n = len(q["tool"]["criteria"]) - 1
            t = self._ask(state, q)
            t_conf = float(t.get("answer_confidence", t["confidence"]))
            t_probs = {k: round(float(v), 4) for k, v in t.get("probabilities", {}).items()}
            if t["choice"] != OTHER and t_conf >= self.tool_threshold:
                tool = t["choice"]
        return RouteResult(
            bucket=bucket_id,
            bucket_confidence=round(b_conf, 4),
            tool=tool,
            tool_confidence=round(t_conf, 4),
            candidates=n,
            ms=round((time.perf_counter() - t0) * 1000, 1),
            bucket_probs={k: round(float(v), 4) for k, v in b.get("probabilities", {}).items()},
            tool_probs=t_probs,
        )
