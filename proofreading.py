"""
proofreading.py — Pluggable proofreading engines for OCR Engine v4.5.0
"""
import os
import time
import logging

log = logging.getLogger(__name__)


class Proofreader:
    """Abstract base class for all proofreading engines."""
    def proofread_lines(self, lines: list[str], context: dict | None = None) -> list[str]:
        raise NotImplementedError


class NoProofreader(Proofreader):
    """Pass-through: returns text unchanged."""
    def proofread_lines(self, lines, context=None):
        return list(lines)


class RuleProofreader(Proofreader):
    """Apply hard-coded correction rules only, no AI."""
    RULES = {
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    }
    def proofread_lines(self, lines, context=None):
        result = []
        for line in lines:
            for bad, good in self.RULES.items():
                line = line.replace(bad, good)
            result.append(line)
        return result


class OllamaProofreader(Proofreader):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434",
                 timeout: int = 60, level: str = "conservative"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.level = level

    def proofread_lines(self, lines, context=None):
        try:
            import urllib.request, json as _json
            prompt = self._build_prompt(lines)
            payload = _json.dumps({"model": self.model, "prompt": prompt,
                                   "stream": False}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = _json.loads(resp.read())
            corrected = data.get("response", "").splitlines()
            return self._validate(lines, corrected)
        except Exception as exc:
            log.warning(f"OllamaProofreader fallback: {exc}")
            return list(lines)

    def _build_prompt(self, lines):
        text = "\n".join(lines)
        return (f"以下是 OCR 掃描的繁體中文文字，共 {len(lines)} 行。"
                f"請校正明顯錯誤，保留每一行結構，不增減行數，不確定時保留原文。\n\n{text}")

    def _validate(self, original, corrected):
        if len(corrected) != len(original):
            log.warning("OllamaProofreader: line count mismatch, reverting")
            return list(original)
        return corrected


class OpenAIProofreader(Proofreader):
    def __init__(self, model: str = "gpt-4o-mini", timeout: int = 60, level: str = "conservative"):
        self.model = model
        self.timeout = timeout
        self.level = level
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    def proofread_lines(self, lines, context=None):
        if not self._api_key:
            return list(lines)
        try:
            import openai
            client = openai.OpenAI(api_key=self._api_key, timeout=self.timeout)
            text = "\n".join(lines)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content":
                    f"校正 OCR 繁體中文，保持 {len(lines)} 行，不確定保留原文：\n{text}"}],
            )
            corrected = resp.choices[0].message.content.splitlines()
            return self._validate(lines, corrected)
        except Exception as exc:
            log.warning(f"OpenAIProofreader fallback: {exc}")
            return list(lines)

    def _validate(self, original, corrected):
        if len(corrected) != len(original):
            return list(original)
        return corrected


class ClaudeProofreader(Proofreader):
    def __init__(self, model: str = "", timeout: int = 60, level: str = "conservative"):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.timeout = timeout
        self.level = level
        self._api_key = (os.environ.get("ANTHROPIC_API_KEY") or
                         os.environ.get("CLAUDE_API_KEY", ""))

    def proofread_lines(self, lines, context=None):
        if not self._api_key:
            return list(lines)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            text = "\n".join(lines)
            msg = client.messages.create(
                model=self.model, max_tokens=4096,
                messages=[{"role": "user", "content":
                    f"校正 OCR 繁體中文，保持 {len(lines)} 行，不確定保留原文：\n{text}"}],
            )
            corrected = msg.content[0].text.splitlines()
            return self._validate(lines, corrected)
        except Exception as exc:
            log.warning(f"ClaudeProofreader fallback: {exc}")
            return list(lines)

    def _validate(self, original, corrected):
        if len(corrected) != len(original):
            return list(original)
        return corrected


class GeminiProofreader(Proofreader):
    def __init__(self, model: str = "gemini-1.5-flash", timeout: int = 60, level: str = "conservative"):
        self.model = model
        self.timeout = timeout
        self.level = level
        self._api_key = os.environ.get("GEMINI_API_KEY", "")

    def proofread_lines(self, lines, context=None):
        if not self._api_key:
            return list(lines)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(self.model)
            text = "\n".join(lines)
            resp = model.generate_content(
                f"校正 OCR 繁體中文，保持 {len(lines)} 行，不確定保留原文：\n{text}")
            corrected = resp.text.splitlines()
            return self._validate(lines, corrected)
        except Exception as exc:
            log.warning(f"GeminiProofreader fallback: {exc}")
            return list(lines)

    def _validate(self, original, corrected):
        if len(corrected) != len(original):
            return list(original)
        return corrected


class OpenAICompatibleProofreader(OpenAIProofreader):
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: int = 60, level: str = "conservative"):
        super().__init__(model=model, timeout=timeout, level=level)
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def proofread_lines(self, lines, context=None):
        if not self._api_key:
            return list(lines)
        try:
            import openai
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url,
                                   timeout=self.timeout)
            text = "\n".join(lines)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content":
                    f"校正 OCR 繁體中文，保持 {len(lines)} 行，不確定保留原文：\n{text}"}],
            )
            corrected = resp.choices[0].message.content.splitlines()
            return self._validate(lines, corrected)
        except Exception as exc:
            log.warning(f"OpenAICompatibleProofreader fallback: {exc}")
            return list(lines)


def create_proofreader(config: dict) -> Proofreader:
    """Factory. config keys: proofreading_engine, proofreading_model, proofreading_level,
    proofreading_base_url, proofreading_api_key."""
    engine = (config.get("proofreading_engine") or "none").lower()
    model  = config.get("proofreading_model", "")
    level  = config.get("proofreading_level", "conservative")
    base_url = config.get("proofreading_base_url", "http://127.0.0.1:11434")

    if engine in ("none", "", "no", "off"):
        return NoProofreader()
    elif engine == "rules":
        return RuleProofreader()
    elif engine == "ollama":
        return OllamaProofreader(model=model or "llama3", base_url=base_url, level=level)
    elif engine in ("openai",):
        return OpenAIProofreader(model=model or "gpt-4o-mini", level=level)
    elif engine in ("claude", "anthropic"):
        log.warning("Claude proofreading is disabled; using offline rule proofreading")
        return RuleProofreader()
    elif engine in ("gemini",):
        return GeminiProofreader(model=model or "gemini-1.5-flash", level=level)
    elif engine in ("openai_compatible", "compatible"):
        return OpenAICompatibleProofreader(
            base_url=config.get("proofreading_base_url", ""),
            model=model, level=level)
    else:
        log.warning(f"Unknown proofreading engine '{engine}', using NoProofreader")
        return NoProofreader()
