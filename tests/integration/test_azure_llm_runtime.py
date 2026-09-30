"""Regression test for invocation-scoped Azure OpenAI construction."""

import sys
import types

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets

from src.graph.graph import Graph
from src.services.llm_runtime import complete_text


_VALUES = {
    "AZURE_OPENAI_API_KEY": "test-key",
    "AZURE_OPENAI_ENDPOINT": "https://test-resource.services.ai.azure.com",
    "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
}


class _RecordingSecrets:
    def __init__(self, requested):
        self._requested = requested

    def require(self, key):
        self._requested.append(key)
        return _VALUES[key]


def _install_recording_client(monkeypatch, configs, calls, content):
    class RecordingClient:
        def __init__(self, config):
            configs.append(config)

        def complete(self, messages):
            calls.append(messages)
            return {"content": content}

    module = types.ModuleType("shared.services.llm.azure_openai_client")
    module.AzureOpenAIClient = RecordingClient
    monkeypatch.setitem(sys.modules, "shared.services.llm.azure_openai_client", module)


def test_azure_client_uses_invocation_scoped_secrets(monkeypatch):
    """BL-13: the client is built per invocation from ctx.secrets, never a global."""
    requested, configs, calls = [], [], []
    _install_recording_client(monkeypatch, configs, calls, "synthetic response")

    state = {
        "correlation_id": "test-correlation",
        "session_id": "test-session",
        "thread_id": "test-thread",
        "trace_id": "test-trace",
        "caller_trust_level": "VERIFIED_EXTERNAL",
    }
    with bound_secrets(_RecordingSecrets(requested)):
        result = complete_text(
            state,
            [{"role": "user", "content": "synthetic prompt"}],
            timeout_s=12.5,
            max_retry=1,
        )

    assert result == "synthetic response"
    assert requested == list(_VALUES)
    assert configs[0]["azure_endpoint"] == _VALUES["AZURE_OPENAI_ENDPOINT"]
    assert configs[0]["timeout"] == 12.5
    assert configs[0]["max_retries"] == 1


def test_marketplace_invocation_reaches_azure(monkeypatch):
    """BL-13: the real Azure path is used for a Marketplace invocation."""
    requested, configs, calls = [], [], []
    _install_recording_client(
        monkeypatch,
        configs,
        calls,
        "## MCP setup\n\nAdd the filesystem server to claude_desktop_config.json.",
    )

    graph = Graph(config={"timeout_s": 12.5, "max_retry": 1, "llm_max_tokens": 600})
    graph.compile()
    assert graph._nodes["post_process"]._llm is None

    with bound_secrets(_RecordingSecrets(requested)):
        result = graph.invoke(
            "How do I set up the filesystem MCP server with Claude Desktop?",
            ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
            input_context={"conversation_history": []},
        )

    assert result["status"] == "success"
    assert "MCP setup" in result["output"]
    assert requested == list(_VALUES)
    assert len(calls) == 1
    assert configs[0]["timeout"] == 12.5
    assert configs[0]["max_retries"] == 1


def test_missing_secrets_return_visible_fallback():
    """BL-14: a missing provider secret degrades, it does not fail the invocation."""

    class MissingSecrets:
        def require(self, key):
            raise KeyError(key)

    graph = Graph(config={"timeout_s": 0.01, "max_retry": 1})
    graph.compile()

    with bound_secrets(MissingSecrets()):
        result = graph.invoke(
            "How do I set up the filesystem MCP server with Claude Desktop?",
            ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
            input_context={"conversation_history": []},
        )

    assert result["status"] == "success"
    assert isinstance(result["output"], str)
    assert "Azure OpenAI synthesis was temporarily unavailable" in result["output"]
