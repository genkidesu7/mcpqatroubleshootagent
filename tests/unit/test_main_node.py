# CMN-C1-069 — Unit Tests: Main Node

import json

from framework.schemas.agent_status import AgentStatus
from src.nodes.main_node import MainNode


def _base_state(**overrides):
    state = {
        "query_id": "q-1",
        "normalized_query": "How do I set up the filesystem MCP server?",
        "intent_class": "setup_config",
        "intent_server_name": "filesystem",
        "server_transport": "stdio",
        "server_category": "filesystem",
        "top_k": 8,
        "score_threshold": 0.6,
    }
    state.update(overrides)
    return state


class _StubRetriever:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.calls = []

    def retrieve(self, query, filters, top_k, score_threshold):
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        if self.error:
            raise self.error
        return self.chunks


class TestMainNodeRetrieval:
    """BL-04 / BL-05 / BL-06: retrieval success, empty, and failure paths."""

    def setup_method(self):
        self.node = MainNode()

    def test_retrieval_success_returns_json_chunks(self):
        retriever = _StubRetriever(
            chunks=[{"chunk_id": "mcp-spec-001", "source": "mcp-spec", "text": "stdio setup", "score": 0.9}]
        )
        state = _base_state(config={"configurable": {"retriever": retriever}})
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieval_count"] == 1
        assert json.loads(result["retrieved_chunks"])[0]["chunk_id"] == "mcp-spec-001"

    def test_intent_and_server_become_retrieval_filters(self):
        retriever = _StubRetriever()
        state = _base_state(config={"configurable": {"retriever": retriever}})
        self.node.execute(state)

        filters = retriever.calls[0]["filters"]
        assert filters["intent_class"] == "setup_config"
        assert filters["server_name"] == "filesystem"
        assert filters["transport"] == "stdio"
        assert filters["category"] == "filesystem"

    def test_unknown_transport_is_not_used_as_a_filter(self):
        retriever = _StubRetriever()
        state = _base_state(
            server_transport="unknown",
            server_category="unknown",
            config={"configurable": {"retriever": retriever}},
        )
        self.node.execute(state)

        filters = retriever.calls[0]["filters"]
        assert "transport" not in filters
        assert "category" not in filters

    def test_no_retriever_returns_no_evidence_rather_than_raising(self):
        """No configured retriever must not invent MCP facts."""
        result = self.node.execute(_base_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieval_count"] == 0
        assert json.loads(result["retrieved_chunks"]) == []

    def test_retrieval_failure_is_recoverable_and_hides_backend_detail(self):
        retriever = _StubRetriever(error=ConnectionError("private backend detail"))
        state = _base_state(config={"configurable": {"retriever": retriever}})
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["recoverable_error_code"] == "retrieval_unavailable"
        assert "private backend detail" not in result["error_message"]

    def test_upstream_input_error_short_circuits(self):
        result = self.node.execute({"input_error_message": "too short"})
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_upstream_error_short_circuits(self):
        result = self.node.execute({"error_message": "boom"})
        assert result["status"] == AgentStatus.ERROR.value


class TestMainNodeErrorDiagnosis:
    """BL-07: known MCP failure patterns are matched without the LLM."""

    def setup_method(self):
        self.node = MainNode()

    def test_401_matches_auth_pattern(self):
        state = _base_state(
            intent_class="error_troubleshoot",
            intent_error_message="error: 401 Unauthorized",
        )
        diagnosis = json.loads(self.node.execute(state)["error_diagnosis"])
        assert diagnosis["matched"] is True
        assert diagnosis["pattern_id"] == "MCP-AUTH-001"
        assert "token" in diagnosis["suggested_fix"].lower()

    def test_connection_refused_matches_transport_pattern(self):
        state = _base_state(
            intent_class="error_troubleshoot",
            intent_error_message="error: connection refused on port 3000",
        )
        diagnosis = json.loads(self.node.execute(state)["error_diagnosis"])
        assert diagnosis["pattern_id"] == "MCP-CONN-001"

    def test_parse_error_matches_protocol_pattern(self):
        state = _base_state(
            intent_class="error_troubleshoot",
            intent_error_message="error: parse error -32700 invalid json",
        )
        diagnosis = json.loads(self.node.execute(state)["error_diagnosis"])
        assert diagnosis["pattern_id"] == "MCP-PROTO-003"
        assert "stdout" in diagnosis["suggested_fix"].lower()

    def test_non_error_intent_produces_no_diagnosis(self):
        state = _base_state(intent_class="setup_config", intent_error_message="error: 401")
        assert json.loads(self.node.execute(state)["error_diagnosis"]) == {}

    def test_unrecognised_error_reports_no_match(self):
        state = _base_state(
            intent_class="error_troubleshoot",
            intent_error_message="error: something entirely unfamiliar happened",
        )
        diagnosis = json.loads(self.node.execute(state)["error_diagnosis"])
        assert diagnosis == {"matched": False}


class TestMainNodeS2SecurityGate:
    """TC-09: PII is masked out of the retrieval query."""

    def test_pii_in_query_is_masked_before_retrieval(self, monkeypatch):
        import src.nodes.main_node as module

        monkeypatch.setattr(
            module,
            "detect_pii",
            lambda text: [
                {
                    "type": "email",
                    "start": text.index("dev@example.com"),
                    "end": text.index("dev@example.com") + len("dev@example.com"),
                }
            ],
        )
        node = MainNode()
        state = {"normalized_query": "My server rejects dev@example.com on auth"}
        gated = node._extra_security_gate_input(state)

        assert "dev@example.com" not in gated["normalized_query"]
        assert "[MASKED]" in gated["normalized_query"]

    def test_clean_query_is_untouched(self, monkeypatch):
        import src.nodes.main_node as module

        monkeypatch.setattr(module, "detect_pii", lambda text: [])
        node = MainNode()
        original = "How do I set up the filesystem MCP server?"
        gated = node._extra_security_gate_input({"normalized_query": original})
        assert gated["normalized_query"] == original


class TestMainNodeAuditLogging:
    """TC-11 / PB-1: a domain event is emitted on every path."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.main_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        MainNode().execute(_base_state())
        assert "main_completed" in events

    def test_retrieval_error_emits_domain_event(self, monkeypatch):
        import src.nodes.main_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        retriever = _StubRetriever(error=TimeoutError("slow"))
        MainNode().execute(_base_state(config={"configurable": {"retriever": retriever}}))
        assert "main_retrieval_error" in events
