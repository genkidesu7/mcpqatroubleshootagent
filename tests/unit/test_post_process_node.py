# CMN-C1-069 — Unit Tests: PostProcess Node

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import PostProcessNode


class _StubLLM:
    def __init__(self, answer="## MCP setup\n\nAdd the server to your config."):
        self.answer = answer
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return {"content": self.answer}


class _FailingLLM:
    def complete(self, messages):
        raise RuntimeError("provider unavailable")


def _chunks(count=2):
    return [
        {
            "chunk_id": f"mcp-spec-{index:03d}",
            "source": "mcp-spec",
            "text": f"Documentation chunk {index}",
            "score": 0.9,
            "spec_version": "2025-06-18",
        }
        for index in range(count)
    ]


def _state(**overrides):
    state = {
        "query_id": "q-1",
        "normalized_query": "How do I set up the filesystem MCP server?",
        "retrieved_chunks": json.dumps(_chunks()),
        "error_diagnosis": "{}",
        "response_budget": 1500,
    }
    state.update(overrides)
    return state


class TestPostProcessSynthesis:
    """BL-08 / BL-09: answer synthesis and graceful degradation."""

    def test_answer_is_synthesized_from_chunks(self):
        node = PostProcessNode(llm=_StubLLM())
        result = node.execute(_state())

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["synthesis_mode"] == "azure_openai"
        assert "MCP setup" in result["answer"]

    def test_diagnosis_is_passed_into_the_prompt(self):
        llm = _StubLLM()
        node = PostProcessNode(llm=llm)
        diagnosis = {
            "matched": True,
            "pattern_id": "MCP-AUTH-001",
            "error_type": "auth",
            "likely_cause": "Invalid token",
            "suggested_fix": "Check the bearer header",
        }
        node.execute(_state(error_diagnosis=json.dumps(diagnosis)))

        user_prompt = llm.calls[0][1]["content"]
        assert "MCP-AUTH-001" in user_prompt
        assert "Check the bearer header" in user_prompt

    def test_provider_failure_degrades_to_evidence_only(self):
        """BL-09: verified documentation is still returned when the provider fails."""
        node = PostProcessNode(llm=_FailingLLM())
        result = node.execute(_state())

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["synthesis_mode"] == "evidence_only"
        assert "mcp-spec-000" in result["answer"]

    def test_provider_failure_with_no_chunks_falls_back_to_diagnosis(self):
        diagnosis = {
            "matched": True,
            "pattern_id": "MCP-CONN-001",
            "error_type": "transport",
            "likely_cause": "Server not running",
            "suggested_fix": "Check the command resolves",
        }
        node = PostProcessNode(llm=_FailingLLM())
        result = node.execute(_state(retrieved_chunks="[]", error_diagnosis=json.dumps(diagnosis)))

        assert result["synthesis_mode"] == "evidence_only"
        assert "MCP-CONN-001" in result["answer"]
        assert "Server not running" in result["answer"]

    def test_no_chunks_and_no_diagnosis_reports_no_results(self):
        node = PostProcessNode(llm=_FailingLLM())
        result = node.execute(_state(retrieved_chunks="[]"))

        assert result["confidence"] == "low"
        assert "No matching MCP documentation found" in result["answer"]

    def test_input_error_is_surfaced_as_guidance(self):
        node = PostProcessNode(llm=_StubLLM())
        result = node.execute({"input_error_message": "too short", "input_error_guidance": ["Name the MCP server"]})

        assert result["status"] == AgentStatus.SUCCESS.value
        payload = json.loads(result["formatted_output"])
        assert payload["status"] == "input_error"
        assert payload["guidance"] == ["Name the MCP server"]

    def test_retrieval_outage_returns_actionable_answer(self):
        node = PostProcessNode(llm=_StubLLM())
        result = node.execute({"recoverable_error_code": "retrieval_unavailable", "trace_id": "tr-9"})

        assert result["status"] == AgentStatus.SUCCESS.value
        assert "temporarily unavailable" in result["answer"].lower()
        assert "tr-9" in result["answer"]

    def test_upstream_error_short_circuits(self):
        node = PostProcessNode(llm=_StubLLM())
        assert node.execute({"error_message": "boom"})["status"] == AgentStatus.ERROR.value


class TestPostProcessOutputShaping:
    """BL-10 / BL-11: snippets, citations, and confidence."""

    def test_code_snippets_are_extracted_with_language(self):
        answer = 'Use this:\n\n```json\n{"mcpServers": {}}\n```\n\nand this:\n\n```ts\nconst c = 1;\n```'
        node = PostProcessNode(llm=_StubLLM(answer=answer))
        result = node.execute(_state())

        snippets = json.loads(result["code_snippets"])
        assert [s["language"] for s in snippets] == ["json", "typescript"]

    def test_citations_are_deduplicated(self):
        duplicated = _chunks(1) * 3
        node = PostProcessNode(llm=_StubLLM())
        result = node.execute(_state(retrieved_chunks=json.dumps(duplicated)))

        assert len(json.loads(result["citations"])) == 1

    @pytest.mark.parametrize(
        "count,score,expected",
        [(6, 0.9, "high"), (3, 0.5, "medium"), (1, 0.9, "low"), (0, 0.0, "low")],
    )
    def test_confidence_tiers(self, count, score, expected):
        chunks = [dict(chunk, score=score) for chunk in _chunks(count)] if count else []
        assert PostProcessNode._compute_confidence(chunks) == expected


class TestPostProcessS3SecurityGate:
    """TC-10 / BL-12: credential redaction preserves documented placeholders."""

    def setup_method(self):
        self.node = PostProcessNode(llm=_StubLLM())

    def test_credential_is_redacted_and_placeholder_preserved(self):
        result = {
            "answer": 'Set the key: api_key = "sk-abcdefghijklmnopqrstuvwxyz123456" or use <your_api_key>.',
            "code_snippets": "[]",
            "formatted_output": json.dumps({"answer": "x", "code_snippets": []}),
        }
        gated = self.node._extra_security_gate_output(result)

        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in gated["answer"]
        assert "[REDACTED]" in gated["answer"]
        assert "<your_api_key>" in gated["answer"]

    @pytest.mark.parametrize(
        "placeholder",
        ["<your_api_key>", "YOUR_API_KEY", "${MCP_TOKEN}", "process.env.MCP_TOKEN", "<<TOKEN>>"],
    )
    def test_documented_placeholders_survive(self, placeholder):
        result = {
            "answer": f"Configure it as {placeholder} in your client config.",
            "code_snippets": "[]",
            "formatted_output": json.dumps({"answer": "x", "code_snippets": []}),
        }
        gated = self.node._extra_security_gate_output(result)
        assert placeholder in gated["answer"]

    def test_credentials_inside_code_snippets_are_redacted(self):
        snippets = [{"snippet_id": "snippet-1", "language": "json", "code": '{"token": "AKIAIOSFODNN7EXAMPLE"}'}]
        result = {
            "answer": "See the config below.",
            "code_snippets": json.dumps(snippets),
            "formatted_output": json.dumps({"answer": "See the config below.", "code_snippets": snippets}),
        }
        gated = self.node._extra_security_gate_output(result)

        assert "AKIAIOSFODNN7EXAMPLE" not in gated["code_snippets"]
        assert "[REDACTED]" in gated["code_snippets"]

    def test_formatted_output_is_kept_in_sync_after_redaction(self):
        result = {
            "answer": 'api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"',
            "code_snippets": "[]",
            "formatted_output": json.dumps({"answer": "stale", "code_snippets": []}),
        }
        gated = self.node._extra_security_gate_output(result)
        assert json.loads(gated["formatted_output"])["answer"] == gated["answer"]

    def test_surviving_credential_blocks_the_output(self, monkeypatch):
        import src.nodes.post_process_node as module

        monkeypatch.setattr(module, "detect_credentials", lambda text: [{"type": "custom"}])
        with pytest.raises(RuntimeError, match="S-3 blocked"):
            self.node._extra_security_gate_output(
                {"answer": "some answer", "code_snippets": "[]", "formatted_output": "{}"}
            )


class TestPostProcessAuditLogging:
    """TC-11 / PB-1: a domain event is emitted on every path."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.post_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PostProcessNode(llm=_StubLLM()).execute(_state())
        assert "post_process_completed" in events

    def test_fallback_path_emits_domain_event(self, monkeypatch):
        import src.nodes.post_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PostProcessNode(llm=_FailingLLM()).execute(_state())
        assert "post_process_llm_fallback" in events
