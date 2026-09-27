"""压缩保真校验层(【本地 PATCH】,借鉴 fast-jev-compaction 思路)。

摘要式压缩会丢硬事实(路径/IP:端口/版本/配置/数字)。本模块在摘要生成后:

1. ``extract_facts`` — 纯正则从被压原文机械抽取硬事实签名(零 LLM 零幻觉,
   只抽原文字符串,不生成不改写);
2. ``check_fidelity`` — 子串校验签名是否仍在摘要中;缺失项一次性调 compression aux
   (qwen3.8-flash, token plan 专用端点,key 读 ~/.hermes/.env 的
   ALIBABA_TOKEN_PLAN_API_KEY,模式同 scripts/ocr_image.py)裁决"是否影响后续任务";
3. ``augment_summary`` — 重要缺失以原文摘录回灌摘要尾部。

fail-open:无缺失不调模型;裁决失败/超时 → important=missing(宁回灌不误删,回灌
上限 MAX_AUGMENT 条,超出丢弃并记 augment_truncated);无 API key → 空转(important=[])。
留痕 logs/compaction_fidelity.log(jsonl,含 verdict_raw 供裁决失败归因)。

抽取噪声治理(2026-09-23 三次真实压缩实证):URL 按端点白名单抽取(本机/内网 +
已在用的 API 域名,文档页/图片/OSS 签名链接不抽);IP 校验段 ≤255 与端口合法性;
无时间分量的纯日期不抽;被过滤的候选整段占位抹除,不泄给后续抽取器。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ENDPOINT = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen3.8-flash"
TIMEOUT = 30
MAX_FACTS = 120
MAX_FACT_LEN = 240
MAX_JUDGED = 80
# 回灌预算:fail-open 全量回灌不设上限曾把 105 条原文摘录灌进摘要尾部(09-23 日志实证)。
MAX_AUGMENT = 10

_SEG = r"[A-Za-z0-9._~+@-]+"
# 字符白名单:旧版排除表漏掉 `\()[]{}*` 等,markdown 残留/字面 \n/中文尾巴整段被吸进 URL。
_RE_URL = re.compile(r"https?://[A-Za-z0-9\-._~:/?#@!$&+=%]+")
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
    # 值排除反斜杠:URL 收紧后残留的字面 \n 碎片(如 \nproviders:)不再被当键值对抽出。
    r"(?:\"[^\"\n]{1,120}\"|'[^'\n]{1,120}'|[^\s\\,;，。；'\"\)\]\}]{1,120})"
)
_RE_NUMID = re.compile(r"\b\d{4,}\b")

# URL 端点白名单(按 hostname 判断):本机/内网地址 + 已在用的 API 端点域名保留;
# 文档页、图片、网页链接不抽(09-23 实证:100+ 条 help.aliyun.com/img.alicdn.com/
# platform.qianwenai.com 等网页 URL 挤爆裁决清单)。宁可白名单外漏抽,不误杀本机端点。
_URL_API_HOSTS = frozenset({
    "api.kimi.com",
    "api.github.com",
    "api.deepseek.com",
    "api.z.ai",
    "api.tikhub.io",
})
# *.aliyuncs.com 只保留 API 形态路径(dashscope/token-plan 的 compatible-mode、/api/、/vN)。
_RE_API_PATH = re.compile(r"^/(?:compatible-mode|api(?:/|$)|v\d)")


def _is_internal_host(host: str) -> bool:
    if host == "localhost":
        return True
    parts = host.split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return False
    first, second = int(parts[0]), int(parts[1])
    return first in (10, 127) or (first == 192 and second == 168) or (first == 172 and 16 <= second <= 31)


def _port_ok(port: Optional[int]) -> bool:
    # 端口合法性 1-65535;单位数端口(:1-:9)在会话文本里几乎从不是真实端点,
    # 是文档碎片/误匹配高发区(09-23 实证 10.0.0.1:1 被当签名回灌),判为碎片。
    return port is None or 10 <= port <= 65535


def _keep_url(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return False
    if not host:
        return False
    if _is_internal_host(host):
        return _port_ok(port)
    # OSS 桶主机名带 .oss- 段,签名链接(Expires/Signature 参数)一次性有效,无回灌价值。
    if ".oss-" in host or host.startswith("oss-"):
        return False
    if host == "aliyuncs.com" or host.endswith(".aliyuncs.com"):
        path = parts.path or "/"
        return path == "/" or bool(_RE_API_PATH.match(path))
    return host in _URL_API_HOSTS


def _tidy(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_url(raw: str) -> str:
    value = _tidy(raw.rstrip(".,;:!?)]}"))
    return value if _keep_url(value) else ""


def _clean_ip(raw: str) -> str:
    match = re.fullmatch(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?::(\d{1,5}))?", raw)
    if not match or any(int(group) > 255 for group in match.groups()[:4]):
        return ""
    port = match.group(5)
    return raw if _port_ok(int(port) if port is not None else None) else ""


def _clean_date(raw: str) -> str:
    value = _tidy(raw)
    # 无时间分量的纯日期不抽(09-23 实证 15+ 条纯日期全是文档噪声);带时分/中文时的保留。
    return value if re.search(r"\d{1,2}:\d{2}|\d\s?时", value) else ""


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
    (_RE_IP, _clean_ip),
    (_RE_DATE, _clean_date),
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
            # 被过滤的候选同样整段占位抹除:垃圾 URL/非法 IP 不泄给路径、版本号等后续抽取器。
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
    # 索引化清单:模型只回编号(几个 token),不再逐字回 echo 长签名(生成量超时病根)。
    numbered = "\n".join(f"[{index}] {fact}" for index, fact in enumerate(missing))
    prompt = (
        f"任务上下文: {hint}\n"
        "以下硬事实在压缩摘要中缺失。请判断哪些对后续任务仍然重要:"
        "文件路径、IP:端口、版本号、命令行、session/进程 ID、关键配置键值通常重要;"
        "泛化的年份、普通日期、与任务无关的编号通常不重要。\n"
        f"{numbered}\n"
        '只返回 JSON：{"important": [编号, ...]}（重要项的编号数组，无重要项返回 []）'
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1000,
        # 关思考链:thinking 耗时随清单长度增长(50 条即 >24s),裁决只需直答,开着必超 30s 读超时。
        "enable_thinking": False,
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
    if any(isinstance(item, int) and not isinstance(item, bool) for item in judged):
        # 索引路径:合法 int 编号回映射原串,越界/非 int 一律丢弃(防幻觉)。
        valid = {
            item
            for item in judged
            if isinstance(item, int) and not isinstance(item, bool) and 0 <= item < len(missing)
        }
        return [fact for index, fact in enumerate(missing) if index in valid], raw
    # 兼容回退:模型回 echo 字符串数组(旧格式)时走原字符串子集匹配。
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
    # 回灌预算:任何分支产出的 important 都在此截断到 MAX_AUGMENT(_judge_important 返回值
    # 与 fail-open 的 list(missing) 同口截断),超出丢弃并记 augment_truncated。
    augment_truncated = max(0, len(important) - MAX_AUGMENT)
    important = important[:MAX_AUGMENT]
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "facts_total": len(facts),
        "facts_hit": len(facts) - len(missing),
        "missing": missing,
        "important": important,
        "augmented": bool(important),
        "augment_truncated": augment_truncated,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "model": MODEL,
        "verdict_ok": verdict_ok,
        "verdict_raw": verdict_raw,
    }
    _write_log(record)
    return {"missing": missing, "important": important, "verdict_raw": verdict_raw}
