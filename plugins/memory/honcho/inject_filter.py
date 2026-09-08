"""Entry filter for Honcho auto-injected memory-context blocks.

【本地 PATCH — 结论库治理双件套①入口过滤（2026-09-08）】
治理对象：honcho 注入块的 "## Explicit Observations" 子节——观测随
Deriver 持续累积，注入时全量上屏，存在三类噪声：
  1) 过期事实（如已卸载的 Vaultwarden 仍被描述为"在用"）
  2) 同一事实多条时间戳重复（如 08-31/09-02/09-03 三次记录同一偏好）
  3) 数量无上限膨胀，挤占 prompt 预算

策略（纯机械、可解释、静默降级）：
  - stale_patterns：用户可编辑的过期正则黑名单（honcho-inject-filter.json）
  - 近重复折叠：正文 char-bigram Jaccard >= 阈值时只保留最新一条
  - 条数上限：只保留最新 N 条，其余折叠并留查询指引

任何异常都返回原文，绝不影响主链路。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / ".hermes" / "honcho-inject-filter.json"

_DEFAULTS = {
    "max_observations": 12,
    "similarity_threshold": 0.55,
    # 已知过期事实（2026-09-08 播种；用户可继续增删）：
    # Vaultwarden 08-31 已卸载、密码自管；"225 passwords" 描述随之作废；dsh 已退役。
    "stale_patterns": [
        r"vaultwarden",
        r"225 (stored )?passwords",
        r"197 (stored )?passwords",
        r"\bdsh\b",
    ],
}

_OBS_HEADING_RE = re.compile(r"^##+\s*Explicit Observations\s*$", re.IGNORECASE)
_ANY_HEADING_RE = re.compile(r"^#")
_OBS_ENTRY_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\]\s*(?P<body>.*)$")


def _load_config() -> dict:
    cfg = dict(_DEFAULTS)
    try:
        if _CONFIG_PATH.exists():
            user_cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            for key in ("max_observations", "similarity_threshold", "stale_patterns"):
                if key in user_cfg:
                    cfg[key] = user_cfg[key]
    except Exception as e:
        logger.debug("honcho inject filter config unreadable (%s); using defaults", e)
    return cfg


def _bigrams(s: str) -> set:
    s = re.sub(r"[^0-9a-zA-Z一-鿿]", "", s.lower())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())


def _jaccard(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def _parse_observations(lines: List[str], start: int) -> Tuple[List[Tuple[str, str]], int]:
    """Parse `[ts] body` entries starting after index start.

    Returns (entries, next_index). Entries are consumed until the next
    heading or a non-entry line that is not a continuation of the current body.
    """
    entries: List[Tuple[str, str]] = []
    i = start
    while i < len(lines):
        line = lines[i]
        m = _OBS_ENTRY_RE.match(line)
        if m:
            entries.append((m.group("ts"), m.group("body")))
            i += 1
            continue
        if not line.strip():
            # blank line: continue only if the next line is another entry
            if i + 1 < len(lines) and _OBS_ENTRY_RE.match(lines[i + 1]):
                i += 1
                continue
            break
        if _ANY_HEADING_RE.match(line):
            break
        # Continuation of the previous entry's body (multi-line observation)
        if entries:
            ts, body = entries[-1]
            entries[-1] = (ts, body + " " + line.strip())
        i += 1
    return entries, i


def _filter_observations(
    entries: List[Tuple[str, str]], cfg: dict
) -> Tuple[List[Tuple[str, str]], int, int]:
    """Drop stale, fold near-dups (keep newest), cap to newest N.

    Returns (kept_entries, dropped_stale, folded_count). Input order is
    preserved (oldest→newest as Honcho renders it); the cap keeps the tail.
    """
    stale_res = []
    for pat in cfg.get("stale_patterns", []):
        try:
            stale_res.append(re.compile(pat, re.IGNORECASE))
        except re.error as e:
            logger.debug("honcho inject filter: bad stale pattern %r (%s)", pat, e)

    kept: List[Tuple[str, str]] = []
    dropped_stale = 0
    for ts, body in entries:
        if any(r.search(body) for r in stale_res):
            dropped_stale += 1
            continue
        kept.append((ts, body))

    # Near-dup fold: later (newer) entry wins. Iterate oldest→newest, keep an
    # entry only if it is not a near-dup of one already kept... that keeps the
    # OLDER copy. Instead dedup against the retained list and replace on hit.
    threshold = float(cfg.get("similarity_threshold", 0.55))
    retained: List[Tuple[str, str]] = []
    folded = 0
    for ts, body in kept:
        dup_idx = next(
            (i for i, (_t, b) in enumerate(retained) if _jaccard(body, b) >= threshold),
            None,
        )
        if dup_idx is not None:
            retained[dup_idx] = (ts, body)  # newer replaces older
            folded += 1
        else:
            retained.append((ts, body))

    max_n = int(cfg.get("max_observations", 12))
    overflow = max(0, len(retained) - max_n)
    retained = retained[-max_n:] if max_n > 0 else retained
    return retained, dropped_stale, folded + overflow


def filter_injection_block(text: str, config_path: Path | None = None) -> str:
    """Filter all Explicit Observations sections in an injected block."""
    if not text or "##" not in text:
        return text
    global _CONFIG_PATH
    if config_path is not None:
        _CONFIG_PATH = Path(config_path)
    cfg = _load_config()

    lines = text.split("\n")
    out: List[str] = []
    i = 0
    total_stale = total_folded = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        if _OBS_HEADING_RE.match(line.strip()):
            entries, end = _parse_observations(lines, i)
            if entries:
                kept, n_stale, n_folded = _filter_observations(entries, cfg)
                total_stale += n_stale
                total_folded += n_folded
                for ts, body in kept:
                    out.append(f"[{ts}] {body}")
                hidden = len(entries) - len(kept)
                if hidden:
                    out.append(
                        f"(… 已折叠 {hidden} 条更早/重复/过期观测；"
                        f"细节用 honcho_search / honcho_context 按需查询)"
                    )
            i = end
    if total_stale or total_folded:
        logger.debug(
            "honcho inject filter: dropped %d stale, folded/capped %d observations",
            total_stale, total_folded,
        )
    return "\n".join(out)
