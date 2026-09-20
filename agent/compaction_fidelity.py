"""压缩保真校验层(【本地 PATCH】,借鉴 fast-jev-compaction 思路)。

摘要式压缩会丢硬事实(路径/IP:端口/版本/配置/数字)。本模块在摘要生成后:

1. ``extract_facts`` — 纯正则从被压原文机械抽取硬事实签名(零 LLM 零幻觉,
   只抽原文字符串,不生成不改写);
2. ``check_fidelity`` — 子串校验签名是否仍在摘要中;缺失项一次性调 compression aux
   (qwen3.8-flash, token plan 专用端点,key 读 ~/.hermes/.env 的
   ALIBABA_TOKEN_PLAN_API_KEY,模式同 scripts/ocr_image.py)裁决"是否影响后续任务";
3. ``augment_summary`` — 重要缺失以原文摘录回灌摘要尾部。

fail-open:无缺失不调模型;裁决失败/超时 → important=missing 全量(宁回灌不误删);
无 API key → 空转(important=[])。留痕 logs/compaction_fidelity.log(jsonl)。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ENDPOINT = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen3.8-flash"
TIMEOUT = 30
MAX_FACTS = 120
MAX_FACT_LEN = 240
MAX_JUDGED = 80

_SEG = r"[A-Za-z0-9._~+@-]+"
_RE_URL = re.compile(r"https?://[^\s<>\"'`（），。；：、”“‘’\]\}】」』]+")
_RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")
_RE_DATE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\b\d{4}/\d{1,2}/\d{1,2}\b"
    r"|\d{4}年\d{1,2}月\d{1,2}日(?:\s?\d{1,2}时(?:\d{1,2}分)?)?"
)
_RE_VERSION = re.compile(r"(?<![\w.])v?\d+\.\d+\.\d+(?:\.\d+)?(?:-[0-9A-Za-z.]+)?(?![\w.])")
_RE_UUID = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
_RE_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_RE_NAMED_ID = re.compile(
    r"(?:session|sess|proc(?:ess)?|pid|tid|task|job|trace|request)[_-]?id\s*[:=]\s*[\"']?[\w.:-]{3,}"
    r"|(?:session|proc|pid|进程|会话|任务)\s*[:=是]?\s*[\"']?[A-Za-z0-9][\w.-]{3,}",
    re.IGNORECASE,
)
_RE_CMD = re.compile(
    r"[A-Za-z_./~][\w./~+@-]*"
    r"(?:[ \t]+[A-Za-z0-9_./~+@=]+)*"
    r"[ \t]+--?[A-Za-z][\w-]*(?:=[^\s`\"'，。；]+)?"
    r"(?:[ \t]+[A-Za-z0-9_./~+@=-]+)*"
)
_RE_PATH = re.compile(
    r"(?<![A-Za-z0-9._~+@-])"
    r"(?:~(?:/" + _SEG + r")+/?"
    r"|\.{1,2}(?:/" + _SEG + r")+/?"
    r"|(?:/" + _SEG + r")+/?"
    r"|(?:[A-Za-z0-9._~+@-]+/){2,}" + _SEG + r")"
)
_RE_KV = re.compile(
    r"(?<![\w./-])[A-Za-z_][\w.-]{1,40}[ \t]*[:=][ \t]*"
    r"(?:\"[^\"\n]{1,120}\"|'[^'\n]{1,120}'|[^\s,;，。；'\"\)\]\}]{1,120})"
)
_RE_NUMID = re.compile(r"\b\d{4,}\b")


def _tidy(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_url(raw: str) -> str:
    return _tidy(raw.rstrip(".,;:!?)]}"))


def _clean_path(raw: str) -> str:
    value = _tidy(raw).rstrip("/")
    if len(value) < 3 or not re.search(r"[A-Za-z._~]", value):
        return ""
    return value


def _clean_kv(raw: str) -> str:
    value = _tidy(raw)
    match = re.match(r"^([\w.-]{2,})\s*[:=]\s*(.+)$", value)
    if not match:
        return ""
    payload = match.group(2).strip("'\"")
    if not payload or not re.search(r"[0-9.=/~_-]", payload):
        return ""
    return value


def _clean_named_id(raw: str) -> str:
    value = _tidy(raw).strip("'\"")
    return value if re.search(r"\d", value) else ""


# 顺序即优先级:先抽的类别整段占位抹除,避免 URL 里的路径、IP 里的版本号被重复抽取。
_EXTRACTORS: Sequence[Tuple[re.Pattern, Callable[[str], str]]] = (
    (_RE_URL, _clean_url),
    (_RE_IP, _tidy),
    (_RE_DATE, _tidy),
    (_RE_VERSION, _tidy),
    (_RE_UUID, _tidy),
    (_RE_LONG_HEX, _tidy),
    (_RE_NAMED_ID, _clean_named_id),
    (_RE_CMD, _tidy),
    (_RE_PATH, _clean_path),
    (_RE_KV, _clean_kv),
    (_RE_NUMID, _tidy),
)


def _mask_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    parts = []
    prev = 0
    for start, end in spans:
        parts.append(text[prev:start])
        parts.append(" " * (end - start))
        prev = end
    parts.append(text[prev:])
    return "".join(parts)


def extract_facts(text: str) -> List[str]:
    """机械抽取硬事实签名,按首次出现顺序去重,只含原文字符串。"""
    if not text:
        return []
    work = text
    facts: List[str] = []
    for pattern, clean in _EXTRACTORS:
        spans = []
        for match in pattern.finditer(work):
            value = clean(match.group(0))
            if value:
                facts.append(value)
                spans.append(match.span())
        work = _mask_spans(work, spans)
    deduped = [fact for fact in dict.fromkeys(facts) if len(fact) <= MAX_FACT_LEN]
    return deduped[:MAX_FACTS]


def texts_from_messages(messages: Sequence[Dict[str, Any]]) -> List[str]:
    """从 OpenAI 消息列表取纯文本(兼容 content 为 str 或多模态 list)。"""
    texts: List[str] = []
    for message in messages:
        content = (message or {}).get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return texts


def augment_summary(summary: str, important_facts: Sequence[str]) -> str:
    """重要缺失签名以原文摘录追加到摘要尾部;为空则原样返回。"""
    if not important_facts:
        return summary
    lines = "\n".join(f"- {fact}" for fact in important_facts)
    return f"{summary}\n\n## ⚠️ 被压内容硬事实（原文摘录，防丢）\n{lines}"


def _load_key() -> str:
    candidates = []
    try:
        from hermes_constants import get_hermes_home

        candidates.append(os.path.join(get_hermes_home(), ".env"))
    except Exception:
        pass
    candidates.append(os.path.expanduser("~/.env"))
    for path in candidates:
        try:
            if os.path.exists(path):
                match = re.search(
                    r"^ALIBABA_TOKEN_PLAN_API_KEY=(.*)$",
                    open(path, encoding="utf-8").read(),
                    re.M,
                )
                if match:
                    return match.group(1).strip()
        except OSError:
            continue
    return os.environ.get("ALIBABA_TOKEN_PLAN_API_KEY", "")


def _log_path() -> str:
    try:
        from hermes_constants import get_hermes_home

        return os.path.join(get_hermes_home(), "logs", "compaction_fidelity.log")
    except Exception:
        return os.path.expanduser("~/.hermes/logs/compaction_fidelity.log")


def _write_log(record: Dict[str, Any]) -> None:
    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _judge_important(key: str, missing: Sequence[str], context_hint: str) -> Tuple[List[str], str]:
    hint = _tidy(context_hint)[:200] or "(无)"
    prompt = (
        f"任务上下文: {hint}\n"
        "以下硬事实签名抽自被压缩的原文,但未出现在压缩摘要里。判断哪些会影响后续任务继续执行:"
        "文件路径、IP:端口、版本号、命令行、session/进程 ID、关键配置键值通常重要;"
        "泛化的年份、普通日期、与任务无关的编号通常不重要。\n"
        '只返回 JSON {"important": [...]},必须是清单的严格子集,签名字符串原样返回,不得改写。\n'
        "清单:\n" + json.dumps(list(missing), ensure_ascii=False)
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1000,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data = json.loads(response.read())
    raw = (data["choices"][0]["message"].get("content") or "").strip()
    judged: List[Any] = []
    block = re.search(r"\{.*\}", raw, re.S)
    if block:
        try:
            parsed = json.loads(block.group(0))
            if isinstance(parsed.get("important"), list):
                judged = parsed["important"]
        except (ValueError, AttributeError):
            judged = []
    picked = {str(item) for item in judged}
    return [fact for fact in missing if fact in picked], raw


def check_fidelity(
    summary: str,
    original_texts: Sequence[str],
    context_hint: str = "",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """校验摘要是否保住原文硬事实签名;缺失的重要签名交模型裁决,fail-open。"""
    started = time.monotonic()
    facts = extract_facts("\n".join(text for text in original_texts if text))
    missing = [fact for fact in facts if fact not in summary]
    important: List[str] = []
    verdict_raw: Optional[str] = None
    verdict_ok = False
    if missing:
        key = _load_key()
        if not key:
            verdict_raw = "no_api_key"
        else:
            judged, overflow = missing[:MAX_JUDGED], missing[MAX_JUDGED:]
            try:
                important, verdict_raw = _judge_important(key, judged, context_hint)
                important.extend(overflow)
                verdict_ok = True
            except Exception as exc:
                # fail-open: 宁回灌不误删
                important = list(missing)
                verdict_raw = f"{type(exc).__name__}: {exc}"[:300]
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "facts_total": len(facts),
        "facts_hit": len(facts) - len(missing),
        "missing": missing,
        "important": important,
        "augmented": bool(important),
        "duration_ms": int((time.monotonic() - started) * 1000),
        "model": MODEL,
        "verdict_ok": verdict_ok,
    }
    _write_log(record)
    return {"missing": missing, "important": important, "verdict_raw": verdict_raw}
