"""Pre-process node for McpQaTroubleshootAgent.

Sanitizes the developer query, then extracts MCP intent: question class,
target server name, SDK client, auth type, and the server transport/category.
Blocks MCP auth-bypass and unsafe tool-call-injection attempts (S-2).
"""

from __future__ import annotations

import re
import uuid
from typing import Any, ClassVar

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event


_INJECTION_PATTERNS = [
    re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<[^>]+>"),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
]

# S-2: attempts to bypass MCP authentication or generate unsafe tool calls.
# Every quantifier is bounded — an unbounded one risks quadratic-time
# backtracking on hostile input.
_ABUSE_PATTERNS = [
    re.compile(r"bypass\s{1,4}(?:\w{1,20}\s{1,4}){0,3}auth(?:entication)?", re.IGNORECASE),
    re.compile(r"disable\s{1,4}(?:mcp\s{1,4}){0,1}auth", re.IGNORECASE),
    re.compile(r"skip\s{1,4}(?:mcp\s{1,4}){0,1}auth", re.IGNORECASE),
    re.compile(r"remove\s{1,4}(?:mcp\s{1,4}){0,1}(?:auth|authentication|security)", re.IGNORECASE),
    re.compile(r"ignore\s{1,4}(?:mcp\s{1,4}){0,1}(?:auth|token|credentials?)", re.IGNORECASE),
    re.compile(r"generate\s{1,4}(?:unsafe|malicious)\s{1,4}tool\s{1,4}call", re.IGNORECASE),
    re.compile(r"inject\s{1,4}(?:mcp\s{1,4}){0,1}(?:tool|command)", re.IGNORECASE),
    re.compile(r"mcp\s{1,4}tool\s{1,4}call\s{1,4}injection", re.IGNORECASE),
    re.compile(r"exploit\s{1,4}mcp", re.IGNORECASE),
    re.compile(r"mcp\s{1,4}(?:server\s{1,4}){0,1}vulnerability", re.IGNORECASE),
    re.compile(r"unauthorized\s{1,4}mcp\s{1,4}tool", re.IGNORECASE),
]

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "setup_config": [
        "set up",
        "setup",
        "install",
        "configure",
        "configuration",
        "config",
        "getting started",
        "initialize",
        "deploy",
        "launch",
        "enable",
        "register",
    ],
    "authentication": [
        "auth",
        "authentication",
        "authorize",
        "authorization",
        "token",
        "api key",
        "api-key",
        "apikey",
        "credentials",
        "secret",
        "permission",
        "bearer",
        "oauth",
    ],
    "error_troubleshoot": [
        "error",
        "exception",
        "traceback",
        "failed",
        "failure",
        "not working",
        "broken",
        "crash",
        "debug",
        "troubleshoot",
        "problem",
        "issue",
        "fix",
        "resolve",
        "cannot",
        "unable",
        "timeout",
        "connection refused",
    ],
    "schema_protocol": [
        "schema",
        "protocol",
        "spec",
        "specification",
        "format",
        "structure",
        "jsonrpc",
        "json-rpc",
        "tool definition",
        "tool schema",
        "resource",
        "prompt",
        "sampling",
        "transport",
        "stdio",
        "sse",
        "capability",
        "negotiation",
    ],
    "best_practice": [
        "best practice",
        "recommend",
        "pattern",
        "design",
        "architecture",
        "best way",
        "optimal",
        "proper",
        "correct way",
        "convention",
        "guidance",
    ],
}

_COMPILED_INTENT: dict[str, list[re.Pattern[str]]] = {
    intent: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords]
    for intent, keywords in _INTENT_KEYWORDS.items()
}

_DEFAULT_INTENT = "schema_protocol"

_TRANSPORT_KEYWORDS: dict[str, list[str]] = {
    "stdio": ["stdio", "standard input", "stdin", "stdout", "subprocess", "local process", "claude desktop"],
    "http": ["http", "https", "rest", "webhook", "streamable http", "remote server"],
    "sse": ["sse", "server-sent events", "server sent events", "event stream", "eventsource"],
}

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "filesystem": ["filesystem", "file system", "directory", "folder", "read file", "write file", "google drive"],
    "database": ["database", "postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis", "query", "table"],
    "api": ["api", "rest api", "endpoint", "external api", "third-party", "brave search", "github", "slack"],
    "custom": ["custom server", "proprietary", "internal server", "build my own", "write my own", "custom mcp"],
}

_COMPILED_TRANSPORT: dict[str, list[re.Pattern[str]]] = {
    key: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords]
    for key, keywords in _TRANSPORT_KEYWORDS.items()
}

_COMPILED_CATEGORY: dict[str, list[re.Pattern[str]]] = {
    key: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords]
    for key, keywords in _CATEGORY_KEYWORDS.items()
}

# Well-known servers resolve transport/category directly rather than by keyword score.
_SERVER_TYPE_MAP: dict[str, tuple[str, str]] = {
    "filesystem": ("stdio", "filesystem"),
    "git": ("stdio", "filesystem"),
    "memory": ("stdio", "custom"),
    "sequential-thinking": ("stdio", "custom"),
    "everything": ("stdio", "custom"),
    "postgres": ("stdio", "database"),
    "sqlite": ("stdio", "database"),
    "puppeteer": ("stdio", "custom"),
    "brave-search": ("http", "api"),
    "fetch": ("http", "api"),
    "gdrive": ("http", "filesystem"),
    "slack": ("http", "api"),
    "github": ("http", "api"),
}

_KNOWN_SERVERS = list(_SERVER_TYPE_MAP) + ["database"]

_SDK_CLIENT_MAP: dict[str, list[str]] = {
    "claude-desktop": ["claude desktop", "claude-desktop"],
    "cursor": ["cursor"],
    "copilot": ["github copilot", "copilot"],
    "vscode": ["vs code", "vscode"],
}

_AUTH_TYPE_KEYWORDS: dict[str, list[str]] = {
    "oauth": ["oauth", "oauth2", "bearer token", "access token"],
    "api-key": ["api key", "api-key", "apikey", "x-api-key"],
    "none": ["no auth", "unauthenticated", "without auth", "no authentication"],
}

_ERROR_INDICATORS = [
    re.compile(r"error[:\s]{1,3}(.{1,400}?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"exception[:\s]{1,3}(.{1,400}?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"\b\d{3}\b[^\n]{0,200}(?:error|failed|denied|unauthorized|forbidden)", re.IGNORECASE),
    re.compile(r"(?:failed|failure)[:\s]{1,3}(.{1,400}?)(?:\n|$)", re.IGNORECASE),
]

_UNKNOWN = "unknown"


class PreProcessNode(FunctionNode):
    """Sanitize the query and extract MCP intent."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("pre_process_skipped", {"reason": "upstream_error"}, result)
            return result

        raw_query = str(state.get("raw_query") or state.get("user_input") or "")
        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        max_input_length = int(state.get("max_input_length") or agent_config.get("max_input_length", 1000))

        if len(raw_query) > max_input_length:
            result = self._input_error(
                raw_query,
                f"The question is too long ({len(raw_query)} characters). The maximum is {max_input_length} characters.",
            )
            emit_trace_event("pre_process_rejected", {"reason": "input_too_long"}, result)
            return result

        normalized_query = self._sanitize(raw_query)
        if not normalized_query.strip():
            if raw_query.strip():
                result = {
                    "raw_query": raw_query,
                    "error_message": "S-1 violation: query is empty after sanitization",
                    "status": AgentStatus.ERROR.value,
                }
            else:
                result = self._input_error(raw_query, "No usable MCP question was provided.")
            emit_trace_event("pre_process_rejected", {"reason": "empty_after_sanitization"}, result)
            return result

        if len(normalized_query) < 10:
            result = self._input_error(
                normalized_query,
                "The question is too short to identify an MCP server, transport, or error.",
            )
            emit_trace_event("pre_process_rejected", {"reason": "input_too_short"}, result)
            return result

        query_id = str(uuid.uuid4())
        intent_server_name = self._extract_server_name(normalized_query)
        intent_error_message = self._extract_error_message(normalized_query)
        intent_sdk_client = self._extract_sdk_client(normalized_query)
        intent_auth_type = self._extract_auth_type(normalized_query)
        intent_class = self._classify_intent(normalized_query, intent_error_message)
        transport, category = self._classify_server_type(normalized_query, intent_server_name)

        emit_trace_event(
            "pre_process_completed",
            {
                "query_id": query_id,
                "intent_class": intent_class,
                "intent_server_name": intent_server_name,
                "server_transport": transport,
                "server_category": category,
            },
            state,
        )

        return {
            "raw_query": raw_query,
            "query_id": query_id,
            "normalized_query": normalized_query,
            "intent_class": intent_class,
            "intent_server_name": intent_server_name,
            "intent_error_message": intent_error_message,
            "intent_sdk_client": intent_sdk_client,
            "intent_auth_type": intent_auth_type,
            "server_transport": transport,
            "server_category": category,
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2: block MCP auth-bypass and unsafe tool-call-injection requests."""
        query = str(state.get("raw_query") or state.get("user_input") or "")
        if any(pattern.search(query) for pattern in _ABUSE_PATTERNS):
            raise SecurityViolationError("PreProcessNode: blocked MCP auth-bypass / unsafe-tool-call-injection pattern")
        return state

    @staticmethod
    def _input_error(raw_query: str, message: str) -> dict[str, Any]:
        return {
            "raw_query": raw_query,
            "error_message": message,
            "input_error_message": message,
            "input_error_guidance": [
                'Name the MCP server, for example: How do I configure the "filesystem" server?',
                "Include the transport (stdio, http, sse), the SDK client, or the exact error text if you have one.",
            ],
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _sanitize(text: str) -> str:
        result = text
        for pattern in _INJECTION_PATTERNS:
            result = pattern.sub("", result)
        return " ".join(result.split())

    @staticmethod
    def _extract_server_name(query: str) -> str | None:
        lower = query.lower()
        for server in _KNOWN_SERVERS:
            if server in lower:
                return server
        if "mcp server" in lower or "mcp-server" in lower:
            return "generic"
        return None

    @staticmethod
    def _extract_error_message(query: str) -> str | None:
        for pattern in _ERROR_INDICATORS:
            match = pattern.search(query)
            if match:
                return match.group(0)[:500]
        return None

    @staticmethod
    def _extract_sdk_client(query: str) -> str | None:
        lower = query.lower()
        for canonical, keywords in _SDK_CLIENT_MAP.items():
            if any(kw in lower for kw in keywords):
                return canonical
        return None

    @staticmethod
    def _extract_auth_type(query: str) -> str | None:
        lower = query.lower()
        for auth_type, keywords in _AUTH_TYPE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return auth_type
        return None

    @staticmethod
    def _classify_intent(query: str, error_message: str | None) -> str:
        text = f"{query} {error_message}" if error_message else query
        scores = {
            intent: sum(1 for pattern in patterns if pattern.search(text))
            for intent, patterns in _COMPILED_INTENT.items()
        }
        # A query carrying an error is a troubleshooting query regardless of
        # whatever else it mentions — this is the highest-cost misclassification.
        if scores["error_troubleshoot"] > 0:
            return "error_troubleshoot"
        best = max(scores, key=lambda key: scores[key])
        return best if scores[best] > 0 else _DEFAULT_INTENT

    @staticmethod
    def _classify_server_type(query: str, server_name: str | None) -> tuple[str, str]:
        if server_name and server_name in _SERVER_TYPE_MAP:
            return _SERVER_TYPE_MAP[server_name]
        transport = PreProcessNode._best_match(query, _COMPILED_TRANSPORT)
        category = PreProcessNode._best_match(query, _COMPILED_CATEGORY)
        return transport, category

    @staticmethod
    def _best_match(text: str, compiled: dict[str, list[re.Pattern[str]]]) -> str:
        scores = {key: sum(1 for pattern in patterns if pattern.search(text)) for key, patterns in compiled.items()}
        best = max(scores, key=lambda key: scores[key])
        return best if scores[best] > 0 else _UNKNOWN
