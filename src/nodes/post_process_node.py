"""Post-process node for McpQaTroubleshootAgent.

Synthesizes the developer answer from retrieved MCP knowledge-base chunks and
the error diagnosis, extracts code snippets and citations, and applies the S-3
output gate.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security.pii_masking import mask_pii
from shared.security.credential_detector import detect_credentials
from shared.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_runtime import complete_text


def _emit_progress(message: str, metadata: dict[str, Any]) -> None:
    """Emit Marketplace progress when the platform event transport is available."""
    try:
        from shared.services.events import emitter
        from shared.services.events.types import EventType
    except (ImportError, ModuleNotFoundError):
        # Lightweight local AgentCore installations may omit Marketplace events.
        # Synthesis and its terminal fallback must remain available regardless.
        return
    emitter().emit_event(
        event_type=EventType.PROGRESS_UPDATE,
        message=message,
        metadata=metadata,
    )


_SYSTEM_PROMPT = (
    "You are an MCP (Model Context Protocol) integration specialist helping a developer.\n"
    "Given a developer question and retrieved MCP specification / SDK documentation, write a "
    "clear, accurate answer.\n\n"
    "Your response MUST:\n"
    "1. Directly answer the question: what to configure, how to authenticate, or what caused the error\n"
    "2. Include runnable code or configuration when the context provides it, in fenced blocks "
    "with a language tag\n"
    "3. Cite each source as: [Source: <source>, Chunk: <chunk_id>]\n"
    "4. Use only the provided context -- never invent MCP methods, fields, or config keys\n"
    "5. Say so explicitly when the context is insufficient\n"
    "6. Use placeholders such as <your_api_key> for secrets -- never emit a real credential\n\n"
    "Keep the answer concise and suitable for a developer reading it mid-task."
)

_NO_RESULTS_ANSWER = (
    "## No matching MCP documentation found\n\n"
    "No MCP specification or SDK documentation matching your question was found. "
    "Please verify the server name, transport (stdio, http, sse), or SDK client and try again.\n\n"
    "_Azure OpenAI synthesis was temporarily unavailable; this response is based on retrieval status only._"
)

_RETRIEVAL_UNAVAILABLE_ANSWER = (
    "## MCP knowledge base temporarily unavailable\n\n"
    "I couldn't reach the MCP documentation index, so I can't safely confirm setup steps, "
    "schema details, or error causes right now. Please try again shortly. If the problem "
    "continues, contact support and include the reference below."
)

# S-3: credential shapes to redact from generated answers and code snippets.
_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI API key", re.compile(r"sk-[a-zA-Z0-9]{20,64}")),
    ("AWS access key", re.compile(r"AKIA[A-Z0-9]{16}")),
    ("JWT token", re.compile(r"eyJ[a-zA-Z0-9._-]{40,512}")),
    ("Bearer token", re.compile(r"Bearer\s{1,3}[a-zA-Z0-9._\-]{20,256}")),
    (
        "Private key block",
        re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]{1,4096}?-----END (?:RSA |EC )?PRIVATE KEY-----"),
    ),
    (
        "Assigned secret",
        re.compile(
            r"(?:api[_-]?key|secret|password|passwd|token)\s{0,4}[=:]\s{0,4}[\"'][a-zA-Z0-9/+_\-]{16,256}[\"']",
            re.IGNORECASE,
        ),
    ),
]

# Documented placeholders are instructional, not secrets. Redacting them would
# destroy the value of a configuration example, which is this agent's purpose.
_PLACEHOLDER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<your[_-][a-zA-Z0-9_-]{1,64}>", re.IGNORECASE),
    re.compile(r"YOUR_[A-Z0-9_]{1,64}"),
    re.compile(r"\$\{[A-Z_]{1,64}\}"),
    re.compile(r"\$[A-Z_]{2,64}"),
    re.compile(r"<<[A-Z_]{1,64}>>"),
    re.compile(r"process\.env\.[A-Z_]{1,64}"),
    re.compile(r"\*{3,}"),
    re.compile(r"xxx+", re.IGNORECASE),
    re.compile(r"INSERT[_\s](?:YOUR[_\s])?(?:API[_\s])?(?:KEY|TOKEN|SECRET)[_\s]HERE", re.IGNORECASE),
]

_REDACTED_MARKER = "[REDACTED]"

_FENCE_PATTERN = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]{0,20})[ \t]{0,8}\n(?P<code>[\s\S]{1,8000}?)```")

_LANG_ALIASES: dict[str, str] = {
    "ts": "typescript",
    "typescript": "typescript",
    "js": "javascript",
    "javascript": "javascript",
    "py": "python",
    "python": "python",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "sh": "shell",
    "bash": "shell",
    "shell": "shell",
    "": "unknown",
}

# The shared name detector favors recall and flags consecutive Title Case words
# as a possible person. These headings are generated by this agent and are not
# personal data.
_NON_PERSON_NAME_ALLOWLIST = frozenset(
    {
        "Model Context Protocol",
        "Claude Desktop",
        "Bearer Token",
        "Likely Cause",
        "Suggested Fix",
        "Error Diagnosis",
    }
)


def _is_placeholder(value: str) -> bool:
    return any(pattern.search(value) for pattern in _PLACEHOLDER_PATTERNS)


def _redact_credentials(text: str) -> tuple[str, list[str]]:
    """Redact real credential shapes; preserve documented placeholders."""
    redacted_items: list[str] = []

    for label, pattern in _CREDENTIAL_PATTERNS:

        def _replacer(match: re.Match[str], _label: str = label) -> str:
            matched = match.group(0)
            if _is_placeholder(matched):
                return matched
            # Record only a short prefix — the audit trail must never carry the
            # full secret.
            redacted_items.append(f"{_label}: {matched[:12]}...")
            return _REDACTED_MARKER

        text = pattern.sub(_replacer, text)

    return text, redacted_items


class PostProcessNode(FunctionNode):
    """Generate the final answer, code snippets, citations, and confidence."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm: Any | None = None,
        *,
        llm_timeout_s: float = 30.0,
        llm_max_retry: int = 3,
    ) -> None:
        super().__init__()
        # Unit-test seam only. Production resolves Azure OpenAI from the
        # invocation-scoped secrets declared by the agent manifest.
        self._llm = llm
        self._llm_timeout_s = llm_timeout_s
        self._llm_max_retry = llm_max_retry

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("input_error_message"):
            message = str(state["input_error_message"])
            guidance = state.get("input_error_guidance", [])
            return {
                "answer": message,
                "citations": "[]",
                "code_snippets": "[]",
                "confidence": "low",
                "formatted_output": json.dumps(
                    {"status": "input_error", "message": message, "guidance": guidance},
                    ensure_ascii=False,
                ),
                "status": AgentStatus.SUCCESS.value,
            }

        if state.get("recoverable_error_code") == "retrieval_unavailable":
            answer = _RETRIEVAL_UNAVAILABLE_ANSWER
            trace_id = str(state.get("trace_id") or "").strip()
            if trace_id:
                answer = f"{answer}\n\nReference: `{trace_id}`"
            emit_trace_event(
                "post_process_retrieval_unavailable",
                {"query_id": state.get("query_id"), "trace_id": trace_id},
                state,
            )
            return {
                "answer": answer,
                "citations": "[]",
                "code_snippets": "[]",
                "confidence": "low",
                "formatted_output": json.dumps(
                    {
                        "status": "temporarily_unavailable",
                        "answer": answer,
                        "citations": [],
                        "code_snippets": [],
                        "confidence": "low",
                        "retrieval_count": 0,
                    },
                    ensure_ascii=False,
                ),
                "error_message": None,
                "recoverable_error_code": None,
                "status": AgentStatus.SUCCESS.value,
            }

        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("post_process_skipped", {"reason": "upstream_error"}, result)
            return result

        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        llm_max_tokens = int(state.get("response_budget") or agent_config.get("llm_max_tokens", 1500))

        chunks = self._load_json_list(state.get("retrieved_chunks"))
        diagnosis = self._load_json_dict(state.get("error_diagnosis"))
        query = str(state.get("normalized_query") or state.get("raw_query") or "")

        if not chunks:
            return self._no_results(state, query, diagnosis, llm_max_tokens)

        context_text = self._build_context(chunks)
        user_prompt = f"Question: {query}\n"
        if diagnosis.get("matched"):
            user_prompt += (
                f"\nA known MCP failure pattern matched this error:\n"
                f"- Pattern: {diagnosis.get('pattern_id')} ({diagnosis.get('error_type')})\n"
                f"- Likely cause: {diagnosis.get('likely_cause')}\n"
                f"- Suggested fix: {diagnosis.get('suggested_fix')}\n"
                "Incorporate this diagnosis into your answer.\n"
            )
        user_prompt += f"\n---\nMCP Documentation:\n{context_text}"

        _emit_progress(
            "Synthesizing the MCP answer with Azure OpenAI...",
            {"stage": "llm_synthesis", "retrieval_count": len(chunks)},
        )
        try:
            answer = self._generate(
                state=state,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=llm_max_tokens,
            )
            synthesis_mode = "azure_openai"
            _emit_progress(
                "Azure OpenAI MCP synthesis completed.",
                {"stage": "llm_synthesis", "status": "completed", "retrieval_count": len(chunks)},
            )
        except Exception as exc:
            emit_trace_event(
                "post_process_llm_fallback",
                {"query_id": state.get("query_id"), "reason": type(exc).__name__},
                state,
            )
            answer = self._fallback_answer(chunks, diagnosis)
            synthesis_mode = "evidence_only"
            _emit_progress(
                "Azure OpenAI synthesis is temporarily unavailable. Returning retrieved documentation only.",
                {"stage": "llm_synthesis", "status": "error", "retrieval_count": len(chunks)},
            )

        code_snippets = self._extract_code_snippets(answer)
        citations = self._extract_citations(chunks)
        confidence = self._compute_confidence(chunks)

        emit_trace_event(
            "post_process_completed",
            {
                "query_id": state.get("query_id"),
                "retrieval_count": len(chunks),
                "citations_count": len(citations),
                "snippet_count": len(code_snippets),
                "confidence": confidence,
            },
            state,
        )

        formatted_output = json.dumps(
            {
                "answer": answer,
                "citations": citations,
                "code_snippets": code_snippets,
                "confidence": confidence,
                "retrieval_count": len(chunks),
                "synthesis_mode": synthesis_mode,
            },
            ensure_ascii=False,
        )
        return {
            "answer": answer,
            "citations": json.dumps(citations),
            "code_snippets": json.dumps(code_snippets),
            "confidence": confidence,
            "synthesis_mode": synthesis_mode,
            "formatted_output": formatted_output,
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def _no_results(
        self,
        state: dict[str, Any],
        query: str,
        diagnosis: dict[str, Any],
        llm_max_tokens: int,
    ) -> dict[str, Any]:
        """No documentation retrieved: still answer from the diagnosis if one matched."""
        _emit_progress(
            "Generating the MCP response with Azure OpenAI...",
            {"stage": "llm_synthesis", "retrieval_count": 0},
        )
        prompt = (
            f"Question: {query}\n\n"
            "No MCP documentation was retrieved. Explain this limitation concisely and suggest "
            "which server name, transport, or SDK client the developer should verify."
        )
        if diagnosis.get("matched"):
            prompt += (
                f"\n\nA known MCP failure pattern did match the reported error:\n"
                f"- Pattern: {diagnosis.get('pattern_id')} ({diagnosis.get('error_type')})\n"
                f"- Likely cause: {diagnosis.get('likely_cause')}\n"
                f"- Suggested fix: {diagnosis.get('suggested_fix')}\n"
                "Lead with this diagnosis, and be clear it comes from the error catalogue "
                "rather than from retrieved documentation."
            )
        try:
            answer = self._generate(
                state=state,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=min(llm_max_tokens, 400),
            )
            synthesis_mode = "azure_openai"
            _emit_progress(
                "Azure OpenAI MCP synthesis completed.",
                {"stage": "llm_synthesis", "status": "completed", "retrieval_count": 0},
            )
        except Exception as exc:
            answer = self._diagnosis_only_answer(diagnosis) if diagnosis.get("matched") else _NO_RESULTS_ANSWER
            synthesis_mode = "evidence_only"
            _emit_progress(
                "Azure OpenAI synthesis is temporarily unavailable. Returning retrieval-only guidance.",
                {"stage": "llm_synthesis", "status": "error", "retrieval_count": 0},
            )
            emit_trace_event("post_process_llm_fallback", {"reason": type(exc).__name__}, state)

        emit_trace_event(
            "post_process_no_results",
            {"query_id": state.get("query_id"), "confidence": "low"},
            state,
        )
        code_snippets = self._extract_code_snippets(answer)
        return {
            "answer": answer,
            "citations": "[]",
            "code_snippets": json.dumps(code_snippets),
            "confidence": "low",
            "synthesis_mode": synthesis_mode,
            "formatted_output": json.dumps(
                {
                    "answer": answer,
                    "citations": [],
                    "code_snippets": code_snippets,
                    "confidence": "low",
                    "retrieval_count": 0,
                    "synthesis_mode": synthesis_mode,
                },
                ensure_ascii=False,
            ),
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _generate(
        self,
        state: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        config = state.get("config")
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        injected = self._llm or configurable.get("llm")
        return complete_text(
            state,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            injected,
            max_tokens=max_tokens,
            timeout_s=self._llm_timeout_s,
            max_retry=self._llm_max_retry,
        )

    @staticmethod
    def _fallback_answer(chunks: list[dict[str, Any]], diagnosis: dict[str, Any]) -> str:
        lines = ["## MCP documentation", "", "LLM synthesis was unavailable; the retrieved documentation follows."]
        if diagnosis.get("matched"):
            lines.extend(
                [
                    "",
                    f"### Error diagnosis — {diagnosis.get('pattern_id')}",
                    f"**Likely cause:** {diagnosis.get('likely_cause')}",
                    f"**Suggested fix:** {diagnosis.get('suggested_fix')}",
                ]
            )
        lines.extend(["", "### Retrieved sources"])
        for chunk in chunks[:20]:
            text = str(chunk.get("text", "")).strip()
            lines.append(f"- [{chunk.get('source', 'unknown')} / {chunk.get('chunk_id', '?')}] " f"{text[:400]}")
        return "\n".join(lines)

    @staticmethod
    def _diagnosis_only_answer(diagnosis: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"## Error diagnosis — {diagnosis.get('pattern_id')}",
                "",
                f"**Likely cause:** {diagnosis.get('likely_cause')}",
                "",
                f"**Suggested fix:** {diagnosis.get('suggested_fix')}",
                "",
                "_No MCP documentation was retrieved and Azure OpenAI synthesis was unavailable; "
                "this diagnosis comes from the known-error catalogue._",
            ]
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        lines = []
        for index, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{index}] Source: {chunk.get('source', 'unknown')} | "
                f"Chunk: {chunk.get('chunk_id', '?')}\n"
                f"    {chunk.get('text', '')}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _extract_code_snippets(answer: str) -> list[dict[str, str]]:
        snippets = []
        for index, match in enumerate(_FENCE_PATTERN.finditer(answer)):
            lang = _LANG_ALIASES.get(match.group("lang").lower(), match.group("lang").lower() or "unknown")
            snippets.append(
                {
                    "snippet_id": f"snippet-{index + 1}",
                    "language": lang,
                    "code": match.group("code").strip(),
                }
            )
        return snippets

    @staticmethod
    def _extract_citations(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        seen = set()
        citations = []
        for chunk in chunks:
            key = (chunk.get("source", ""), chunk.get("chunk_id", ""))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "source": str(chunk.get("source") or ""),
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "spec_version": str(chunk.get("spec_version") or ""),
                }
            )
        return citations

    @staticmethod
    def _compute_confidence(chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "low"
        try:
            scores = [float(chunk.get("score", 0.0)) for chunk in chunks if chunk.get("score")]
            avg_score = sum(scores) / len(scores) if scores else 0.0
        except (ValueError, TypeError):
            avg_score = 0.0
        if len(chunks) >= 5 and avg_score >= 0.75:
            return "high"
        if len(chunks) >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _load_json_list(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, str):
            return []
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []

    @staticmethod
    def _load_json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return {}
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # ------------------------------------------------------------------ #
    # S-3
    # ------------------------------------------------------------------ #

    def _extra_security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """S-3: redact credential shapes, then mask residual PII.

        Redaction happens here, in the node that produces the content, so the
        framework's own @final credential scan sees clean output and the run ends
        in a redacted SUCCESS. A downstream masking node would never be reached:
        this node's gate raises first. Redaction is owned by the producing node.
        """
        answer = result.get("answer", "")
        if not answer:
            return result

        redacted_answer, redacted_items = _redact_credentials(answer)
        snippets = self._load_json_list(result.get("code_snippets"))
        for snippet in snippets:
            code = str(snippet.get("code", ""))
            if code:
                snippet["code"], found = _redact_credentials(code)
                redacted_items.extend(found)

        if redacted_items:
            redacted_answer = f"{redacted_answer}\n\n_Credential-shaped values were redacted from this response._"
            emit_trace_event(
                "post_process_s3_credential_redacted",
                {"redacted_count": len(redacted_items)},
                result,
            )

        # Any credential surviving redaction is a detector gap, not a safe output.
        surviving = detect_credentials(redacted_answer)
        if surviving:
            emit_trace_event(
                "post_process_s3_credential_blocked",
                {"finding_type": surviving[0]["type"]},
                result,
            )
            raise RuntimeError("S-3 blocked: credential detected in output")

        # Redact PII rather than raising. Raising turns a recoverable output-filter
        # decision into a failed invocation, and the shared detector's high-recall
        # name heuristic also matches domain headings such as "Claude Desktop".
        pii_findings = [
            finding
            for finding in detect_pii(redacted_answer)
            if finding.get("type") != "name" or finding.get("value") not in _NON_PERSON_NAME_ALLOWLIST
        ]
        if pii_findings:
            redacted_answer = mask_pii(redacted_answer, pii_findings)
            redacted_answer = f"{redacted_answer}\n\n_Some personal information was redacted from this response._"
            emit_trace_event(
                "post_process_s3_pii_redacted",
                {
                    "finding_count": len(pii_findings),
                    "finding_types": sorted({str(item["type"]) for item in pii_findings}),
                },
                result,
            )

        result["answer"] = redacted_answer
        result["code_snippets"] = json.dumps(snippets, ensure_ascii=False)
        self._update_formatted_output(result, snippets)
        return result

    @staticmethod
    def _update_formatted_output(result: dict[str, Any], snippets: list[dict[str, Any]]) -> None:
        raw = result.get("formatted_output")
        if not isinstance(raw, str):
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        payload["answer"] = result["answer"]
        payload["code_snippets"] = snippets
        result["formatted_output"] = json.dumps(payload, ensure_ascii=False)
