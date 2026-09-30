"""Main node for McpQaTroubleshootAgent.

Retrieves MCP knowledge-base chunks for the parsed query and, for
troubleshooting questions, matches the reported error against a catalogue of
known MCP failure patterns.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event


# Known MCP failure patterns. Each entry yields an actionable diagnosis without
# needing the LLM, so troubleshooting still works when synthesis degrades.
_ERROR_PATTERNS: list[dict[str, Any]] = [
    {
        "pattern_id": "MCP-AUTH-001",
        "regex": re.compile(r"\b401\b|unauthorized|invalid.{0,20}token|invalid.{0,20}api.?key", re.IGNORECASE),
        "error_type": "auth",
        "likely_cause": "Invalid or missing API key / auth token. The MCP server rejected the credential.",
        "suggested_fix": (
            "Verify the API key or token in your MCP client config. For Claude Desktop, check the "
            "`env` block of `claude_desktop_config.json`. For HTTP transport, confirm the "
            "`Authorization: Bearer <token>` header is sent."
        ),
    },
    {
        "pattern_id": "MCP-AUTH-002",
        "regex": re.compile(r"\b403\b|forbidden|permission.{0,5}denied|access.{0,5}denied", re.IGNORECASE),
        "error_type": "auth",
        "likely_cause": "The credential is valid but lacks the permissions required for this operation.",
        "suggested_fix": (
            "Check the scopes attached to the API key, and confirm the MCP server grants the "
            "requested tool or resource to that principal."
        ),
    },
    {
        "pattern_id": "MCP-CONN-001",
        "regex": re.compile(r"connection.{0,5}refused|econnrefused|cannot.{0,10}connect", re.IGNORECASE),
        "error_type": "transport",
        "likely_cause": "The MCP server process is not running, or the configured host/port is wrong.",
        "suggested_fix": (
            "For stdio, confirm the `command` and `args` resolve and the process starts cleanly. "
            "For HTTP/SSE, confirm the URL, port, and that the server is listening."
        ),
    },
    {
        "pattern_id": "MCP-CONN-002",
        "regex": re.compile(r"timeout|timed.{0,3}out|etimedout", re.IGNORECASE),
        "error_type": "transport",
        "likely_cause": "The server accepted the connection but did not respond within the client timeout.",
        "suggested_fix": (
            "Check the server logs for a blocked handler. Long-running tools should stream progress "
            "rather than hold the request open past the client timeout."
        ),
    },
    {
        "pattern_id": "MCP-PROTO-001",
        "regex": re.compile(r"method.{0,5}not.{0,5}found|unknown.{0,5}method|-32601", re.IGNORECASE),
        "error_type": "protocol",
        "likely_cause": "The client called a JSON-RPC method the server does not implement.",
        "suggested_fix": (
            "Confirm the server advertises the capability during initialization, and that the client "
            "and server negotiated a protocol version that includes the method."
        ),
    },
    {
        "pattern_id": "MCP-PROTO-002",
        "regex": re.compile(r"invalid.{0,10}params|-32602|schema.{0,10}valid", re.IGNORECASE),
        "error_type": "protocol",
        "likely_cause": "Tool arguments did not match the tool's declared input schema.",
        "suggested_fix": (
            "Compare the call arguments against the tool's `inputSchema`. Required properties, types, "
            "and enum values must all match."
        ),
    },
    {
        "pattern_id": "MCP-PROTO-003",
        "regex": re.compile(r"parse.{0,5}error|invalid.{0,5}json|-32700", re.IGNORECASE),
        "error_type": "protocol",
        "likely_cause": "Malformed JSON-RPC payload — often stray stdout output on the stdio transport.",
        "suggested_fix": (
            "On stdio, the server must write only JSON-RPC frames to stdout. Route logging to stderr; "
            "a stray `print()` corrupts the stream."
        ),
    },
]


class MainNode(FunctionNode):
    """Retrieve MCP knowledge-base chunks and diagnose reported errors."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__()
        # Required scaffold injection point. Retrieval and diagnosis are
        # deterministic today; response synthesis consumes the client in
        # PostProcessNode.
        self._llm = llm

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("input_error_message"):
            return {"status": AgentStatus.SUCCESS.value}

        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("main_skipped", {"reason": "upstream_error"}, result)
            return result

        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        top_k = int(state.get("top_k") or agent_config.get("top_k", 8))
        score_threshold = float(state.get("score_threshold") or agent_config.get("score_threshold", 0.6))

        normalized_query = str(state.get("normalized_query") or state.get("raw_query") or "")

        filters: dict[str, Any] = {}
        if state.get("intent_class"):
            filters["intent_class"] = state.get("intent_class")
        if state.get("intent_server_name"):
            filters["server_name"] = state.get("intent_server_name")
        if state.get("server_transport") and state.get("server_transport") != "unknown":
            filters["transport"] = state.get("server_transport")
        if state.get("server_category") and state.get("server_category") != "unknown":
            filters["category"] = state.get("server_category")

        try:
            chunks = self._retrieve(
                query=normalized_query,
                filters=filters,
                top_k=top_k,
                score_threshold=score_threshold,
                config=state.get("config"),
            )
        except Exception as exc:
            error_message = f"MainNode: the MCP knowledge-base index is temporarily unavailable ({type(exc).__name__})"
            emit_trace_event(
                "main_retrieval_error",
                {"query_id": state.get("query_id"), "error_type": type(exc).__name__},
                state,
            )
            return {
                "retrieved_chunks": "[]",
                "retrieval_count": 0,
                "error_diagnosis": "{}",
                "error_message": error_message,
                "recoverable_error_code": "retrieval_unavailable",
                # The failure is handled by PostProcessNode so Marketplace receives
                # an actionable response instead of treating the invocation itself
                # as failed.
                "status": AgentStatus.SUCCESS.value,
            }

        diagnosis = self._diagnose(
            intent_class=str(state.get("intent_class") or ""),
            error_message=state.get("intent_error_message"),
        )

        retrieval_count = len(chunks)
        emit_trace_event(
            "main_completed",
            {
                "query_id": state.get("query_id"),
                "retrieval_count": retrieval_count,
                "filters_applied": list(filters.keys()),
                "diagnosis_pattern_id": diagnosis.get("pattern_id"),
            },
            state,
        )

        return {
            "retrieved_chunks": json.dumps(chunks),
            "retrieval_count": retrieval_count,
            "error_diagnosis": json.dumps(diagnosis),
            "error_message": None,
            "recoverable_error_code": None,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _retrieve(
        query: str,
        filters: dict[str, Any],
        top_k: int,
        score_threshold: float,
        config: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        configurable = (config or {}).get("configurable", {})
        retriever = configurable.get("retriever")
        if retriever is not None:
            return cast(
                list[dict[str, Any]],
                retriever.retrieve(
                    query=query,
                    filters=filters,
                    top_k=top_k,
                    score_threshold=score_threshold,
                ),
            )
        # No retriever configured: return no evidence rather than inventing MCP
        # facts. PostProcessNode turns this into an explicit no-results answer.
        return []

    @staticmethod
    def _diagnose(intent_class: str, error_message: str | None) -> dict[str, Any]:
        """Match a reported error against the known MCP failure catalogue."""
        if intent_class != "error_troubleshoot" or not error_message:
            return {}
        for pattern in _ERROR_PATTERNS:
            if pattern["regex"].search(error_message):
                return {
                    "matched": True,
                    "pattern_id": pattern["pattern_id"],
                    "error_type": pattern["error_type"],
                    "likely_cause": pattern["likely_cause"],
                    "suggested_fix": pattern["suggested_fix"],
                }
        return {"matched": False}

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2: PII masking on normalized_query before knowledge-base retrieval."""
        raw = state.get("normalized_query", "")
        if not raw:
            return state
        findings = detect_pii(raw)
        if findings:
            for finding in reversed(findings):
                raw = raw[: finding["start"]] + "[MASKED]" + raw[finding["end"] :]
            state["normalized_query"] = raw
            emit_trace_event("pii_masked_normalized_query", {}, state)
        return state
