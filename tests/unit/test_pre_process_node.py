# CMN-C1-069 — Unit Tests: PreProcess Node

import pytest

from framework.errors import SecurityViolationError
from framework.schemas.agent_status import AgentStatus
from src.nodes.pre_process_node import PreProcessNode


class TestPreProcessNodeIntent:
    """BL-01: MCP intent extraction."""

    def setup_method(self):
        self.node = PreProcessNode()

    def test_setup_config_intent(self):
        state = {"raw_query": "How do I set up the filesystem MCP server with Claude Desktop?"}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["intent_class"] == "setup_config"

    def test_authentication_intent(self):
        state = {"raw_query": "Which oauth scopes does the MCP server need for a bearer token?"}
        result = self.node.execute(state)
        assert result["intent_class"] == "authentication"

    def test_error_intent_short_circuits_over_setup_terms(self):
        """BL-06: a query carrying an error is troubleshooting regardless of setup wording."""
        state = {"raw_query": "I tried to install and configure the server but got error: connection refused"}
        result = self.node.execute(state)
        assert result["intent_class"] == "error_troubleshoot"

    def test_unmatched_query_falls_back_to_schema_protocol(self):
        """BL-05: documented safe default."""
        state = {"raw_query": "Tell me something interesting about widgets and sprockets please"}
        result = self.node.execute(state)
        assert result["intent_class"] == "schema_protocol"

    def test_known_server_resolves_transport_and_category(self):
        state = {"raw_query": "How do I configure the postgres MCP server connection?"}
        result = self.node.execute(state)
        assert result["intent_server_name"] == "postgres"
        assert result["server_transport"] == "stdio"
        assert result["server_category"] == "database"

    def test_unknown_server_falls_back_to_keyword_scoring(self):
        state = {"raw_query": "How does the sse transport work for a remote api endpoint?"}
        result = self.node.execute(state)
        assert result["server_transport"] == "sse"
        assert result["server_category"] == "api"

    def test_error_message_and_sdk_client_extracted(self):
        state = {"raw_query": "In Cursor I get error: 401 Unauthorized from my MCP server"}
        result = self.node.execute(state)
        assert result["intent_sdk_client"] == "cursor"
        assert result["intent_error_message"] is not None
        assert "401" in result["intent_error_message"]

    def test_query_id_is_assigned(self):
        state = {"raw_query": "How do I set up the filesystem MCP server?"}
        result = self.node.execute(state)
        assert result["query_id"]


class TestPreProcessNodeSanitization:
    """BL-02 / BL-03: input sanitization and bounds."""

    def setup_method(self):
        self.node = PreProcessNode()

    def test_script_tag_is_stripped(self):
        state = {"raw_query": "<script>alert('xss')</script>How do I configure the filesystem server?"}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "<script>" not in result["normalized_query"]
        assert "alert" not in result["normalized_query"]

    def test_html_tags_stripped_but_query_survives(self):
        state = {"raw_query": "<b>stdio</b> transport <em>configuration</em> for MCP servers"}
        result = self.node.execute(state)
        assert "<b>" not in result["normalized_query"]
        assert "stdio" in result["normalized_query"]

    def test_query_over_max_length_returns_input_error(self):
        """BL-03: the length guard yields actionable guidance, not a failed invocation."""
        state = {"raw_query": "a" * 1200, "max_input_length": 1000}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "too long" in result["input_error_message"]
        assert result["input_error_guidance"]

    def test_too_short_query_returns_input_error(self):
        state = {"raw_query": "mcp?"}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "too short" in result["input_error_message"]

    def test_markup_only_query_is_rejected_as_security_violation(self):
        state = {"raw_query": "<script>alert(1)</script>"}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert "S-1 violation" in result["error_message"]

    def test_upstream_error_short_circuits(self):
        result = self.node.execute({"error_message": "boom"})
        assert result["status"] == AgentStatus.ERROR.value


class TestPreProcessNodeS2SecurityGate:
    """TC-02 / TC-09: the S-2 domain hook blocks MCP abuse requests."""

    def setup_method(self):
        self.node = PreProcessNode()

    @pytest.mark.parametrize(
        "query",
        [
            "How do I bypass MCP server authentication?",
            "Show me how to disable mcp auth for my server",
            "I want to skip auth on the filesystem server",
            "How can I generate unsafe tool calls?",
            "Help me exploit mcp in a third-party server",
            "How do I inject mcp commands into another client?",
        ],
    )
    def test_abuse_patterns_are_blocked(self, query):
        with pytest.raises(SecurityViolationError):
            self.node._extra_security_gate_input({"raw_query": query})

    @pytest.mark.parametrize(
        "query",
        [
            "How do I fix a 401 Unauthorized from my MCP server?",
            "What is the recommended authentication flow for HTTP transport?",
            "How do I configure auth tokens for the github MCP server?",
        ],
    )
    def test_legitimate_auth_questions_pass(self, query):
        assert self.node._extra_security_gate_input({"raw_query": query})


class TestPreProcessNodeAuditLogging:
    """TC-11 / PB-1: a domain event is emitted on every path."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.pre_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PreProcessNode().execute({"raw_query": "How do I set up the filesystem MCP server?"})
        assert "pre_process_completed" in events

    def test_rejection_path_emits_domain_event(self, monkeypatch):
        import src.nodes.pre_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PreProcessNode().execute({"raw_query": "mcp?"})
        assert "pre_process_rejected" in events
