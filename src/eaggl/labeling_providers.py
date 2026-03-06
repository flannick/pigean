from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelingRequest:
    prompt: str
    auth_key: str
    model: str


class LabelingProvider:
    provider_name = "base"

    def query(self, request: LabelingRequest, warn_fn=None):
        raise NotImplementedError


class OpenAILabelingProvider(LabelingProvider):
    provider_name = "openai"

    def query(self, request: LabelingRequest, warn_fn=None):
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": 0,
        }
        http_request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % request.auth_key,
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=60) as response_fh:
                response_payload = json.loads(response_fh.read().decode("utf-8"))
            choices = response_payload.get("choices", [])
            if len(choices) == 0:
                if warn_fn is not None:
                    warn_fn("OpenAI response missing choices; skipping LLM labels")
                return None
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None:
                if warn_fn is not None:
                    warn_fn("OpenAI response missing message content; skipping LLM labels")
                return None
            return content
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = str(e)
            if warn_fn is not None:
                warn_fn("OpenAI labeling request failed: HTTP %s %s" % (e.code, body))
            return None
        except urllib.error.URLError as e:
            if warn_fn is not None:
                warn_fn("OpenAI labeling request failed: %s" % e)
            return None
        except Exception as e:
            if warn_fn is not None:
                warn_fn("OpenAI labeling request failed: %s" % e)
            return None


def resolve_labeling_provider(provider_name, bail_fn=None):
    provider = (provider_name if provider_name is not None else "openai").strip().lower()
    if provider == "openai":
        return OpenAILabelingProvider()
    if provider == "gemini":
        bail_fn("LLM provider 'gemini' is not implemented yet; use --lmm-provider openai")
    if provider == "claude":
        bail_fn("LLM provider 'claude' is not implemented yet; use --lmm-provider openai")
    bail_fn("Unsupported --lmm-provider '%s'; expected one of: openai, gemini, claude" % provider)
