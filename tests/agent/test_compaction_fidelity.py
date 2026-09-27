"""Tests for agent/compaction_fidelity.py — 抽取噪声治理 + 回灌预算 + 裁决留痕。

覆盖 2026-09-23 三次真实压缩暴露的缺陷(logs/compaction_fidelity.log 实证):
URL 端点白名单(文档页/图片/OSS 签名链接过滤,本机与 API 端点保留)、IP:端口校验
(10.0.0.1:1 类碎片不抽)、无时间分量纯日期不抽、fail-open 回灌截断到 MAX_AUGMENT
且 record 落 verdict_raw / augment_truncated。
"""

from __future__ import annotations

from typing import List

from agent import compaction_fidelity as cf


def _kv_texts(count: int) -> List[str]:
    """每条文本贡献一个可被 KV 抽取器捕获的硬事实。"""
    return [f"配置项 key{i}=value-{i} 记录。" for i in range(count)]


class TestExtractUrlWhitelist:
    def test_filters_web_urls_keeps_api_endpoints(self):
        text = (
            "文档 https://help.aliyun.com/zh/model-studio/error-code#apikey-error 参考,"
            "配图 https://img.alicdn.com/imgextra/i3/O1CN01XHTYyR_!!6000000005719-55-tps-128-28.svg ,"
            "控制台 https://platform.qianwenai.com/home/billing/subscription/token-plan 见。"
            "端点 https://dashscope.aliyuncs.com/compatible-mode/v1 与 "
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 备用,"
            "另有 https://api.kimi.com/coding 和 https://api.github.com/repos/OWNER/REPO 在用。"
        )
        facts = cf.extract_facts(text)
        assert not any("help.aliyun.com" in fact for fact in facts)
        assert not any("alicdn.com" in fact for fact in facts)
        assert not any("6000000005719" in fact for fact in facts)
        assert not any("qianwenai.com" in fact for fact in facts)
        assert "https://dashscope.aliyuncs.com/compatible-mode/v1" in facts
        assert "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" in facts
        assert "https://api.kimi.com/coding" in facts
        assert "https://api.github.com/repos/OWNER/REPO" in facts

    def test_keeps_internal_endpoints_drops_oss_signed_url(self):
        text = (
            "快照 https://dashscope-5859.oss-cn-wulanchabu-acdr-1.aliyuncs.com/1d/a6/20260920/x.png"
            "?Expires=1789972407&OSSAccessKeyId=LTAI5tKPD3TMqf2Lna1fASuh&Signature=Mvhxk3mn%3D 已过期。"
            "本机 http://127.0.0.1:8000/v3/workspaces/hermes/conclusions 与 "
            "http://10.173.48.28:8001/mcp 在用,网关监听 0.0.0.0:8001。"
        )
        facts = cf.extract_facts(text)
        assert "http://127.0.0.1:8000/v3/workspaces/hermes/conclusions" in facts
        assert "http://10.173.48.28:8001/mcp" in facts
        assert "0.0.0.0:8001" in facts
        assert not any("oss-cn-wulanchabu" in fact for fact in facts)
        assert not any("Signature=" in fact or "OSSAccessKeyId" in fact for fact in facts)

    def test_aliyuncs_host_requires_api_path_unless_root(self):
        text = (
            "端点根 http://llm-lkjvkdrlbakjprjp.cn-beijing.maas.aliyuncs.com/ 在用,"
            "接口 https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis 保留。"
        )
        facts = cf.extract_facts(text)
        assert "http://llm-lkjvkdrlbakjprjp.cn-beijing.maas.aliyuncs.com/" in facts
        assert "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis" in facts

    def test_url_stops_at_literal_backslash_and_cjk(self):
        text = (
            "base_url: https://api.kimi.com/coding\\nproviders:\\n 配置,"
            "文档 https://www.qianwenai.com/hub/mcp 不抽。"
        )
        facts = cf.extract_facts(text)
        assert "https://api.kimi.com/coding" in facts
        assert not any("\\n" in fact for fact in facts)
        assert not any("providers" in fact for fact in facts)
        assert not any("qianwenai.com" in fact for fact in facts)


class TestExtractIpValidation:
    def test_drops_fragment_and_invalid_ip_port(self):
        facts = cf.extract_facts(
            "误匹配样本 10.0.0.1:1 不抽,真实端点 10.173.48.28:8001 保留;"
            "非法 999.1.2.3 与越界端口 10.1.2.3:99999 也不抽。"
        )
        assert "10.173.48.28:8001" in facts
        assert "10.0.0.1:1" not in facts
        assert "10.0.0.1" not in facts
        assert not any(fact.startswith("999.") for fact in facts)
        assert not any(fact.endswith(":99999") for fact in facts)


class TestExtractDateFilter:
    def test_drops_pure_date_keeps_datetime(self):
        facts = cf.extract_facts(
            "文档日期 2026-08-17 与 2026/09/02、2026年5月9日 均为噪声;"
            "故障发生在 2026-09-20 19:39:42,复盘于 2026-09-23T07:56:11Z。"
        )
        assert "2026-08-17" not in facts
        assert "2026/09/02" not in facts
        assert "2026年5月9日" not in facts
        assert "2026-09-20 19:39:42" in facts
        assert "2026-09-23T07:56:11Z" in facts


class TestAugmentBudget:
    def test_failopen_augment_capped_and_logged(self, monkeypatch):
        monkeypatch.setattr(cf, "_load_key", lambda: "fake-key")

        def _raise(*args, **kwargs):
            raise TimeoutError("timed out")

        monkeypatch.setattr(cf, "_judge_important", _raise)
        records = []
        monkeypatch.setattr(cf, "_write_log", records.append)

        result = cf.check_fidelity("摘要:不含任何硬事实。", _kv_texts(25), session_id="t-failopen")

        assert result["important"] == result["missing"][: cf.MAX_AUGMENT]
        record = records[-1]
        assert record["verdict_ok"] is False
        assert record["verdict_raw"].startswith("TimeoutError")
        assert record["augment_truncated"] == len(result["missing"]) - cf.MAX_AUGMENT
        assert record["important"] == result["important"]

    def test_judged_important_also_capped(self, monkeypatch):
        monkeypatch.setattr(cf, "_load_key", lambda: "fake-key")
        monkeypatch.setattr(
            cf, "_judge_important", lambda key, missing, hint: (list(missing), '{"important": "all"}')
        )
        records = []
        monkeypatch.setattr(cf, "_write_log", records.append)

        result = cf.check_fidelity("摘要:不含任何硬事实。", _kv_texts(12))

        assert len(result["important"]) == cf.MAX_AUGMENT
        record = records[-1]
        assert record["verdict_ok"] is True
        assert record["verdict_raw"] == '{"important": "all"}'
        assert record["augment_truncated"] == len(result["missing"]) - cf.MAX_AUGMENT


class TestVerdictRawLogging:
    def test_verdict_raw_persisted_on_success(self, monkeypatch):
        monkeypatch.setattr(cf, "_load_key", lambda: "fake-key")
        monkeypatch.setattr(cf, "_judge_important", lambda *args: ([], '{"important": []}'))
        records = []
        monkeypatch.setattr(cf, "_write_log", records.append)

        cf.check_fidelity("摘要。", _kv_texts(3))

        assert records[-1]["verdict_raw"] == '{"important": []}'

    def test_no_missing_skips_judge_and_logs_null_verdict(self, monkeypatch):
        records = []
        monkeypatch.setattr(cf, "_write_log", records.append)

        def _judge(*args):
            raise AssertionError("judge must not run when nothing is missing")

        monkeypatch.setattr(cf, "_judge_important", _judge)
        texts = _kv_texts(3)
        facts = cf.extract_facts("\n".join(texts))
        result = cf.check_fidelity("摘要 " + " ".join(facts), texts)

        assert result["missing"] == []
        assert result["important"] == []
        record = records[-1]
        assert "verdict_raw" in record and record["verdict_raw"] is None
        assert record["augment_truncated"] == 0
