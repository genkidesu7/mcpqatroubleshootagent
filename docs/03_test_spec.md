# Test Specification

## Test Strategy

- Coverage target: 80%
- Test types: unit, framework compliance, proof-of-boundary, integration, and local STG invocation.
- Unit tests may call execute directly for isolated business logic.
- Security-boundary tests call node(state) through FunctionNode.__call__.
- No live Azure OpenAI or vector index is required: the provider is exercised through a recording
  double, and retrieval through an injected stub retriever.

## Framework Compliance Tests

| TC-ID | Test | Expected Result |
|-------|------|-----------------|
| TC-01 | State contract is a flat TypedDict | No Pydantic, dataclass, secret, or runtime client fields |
| TC-02 | Invalid or unsafe input is rejected or sanitized | Safe failure, markup stripped, or masked retrieval query |
| TC-03 | No credential in State | Credential scan reports zero violations |
| TC-04 | InvocationContext construction inside nodes | Only from_state is allowed; authenticated server adapter is exempt |
| TC-05 | No duplicate lifecycle trace events | execute methods do not emit node_start, node_complete, or node_error |
| TC-06 | FunctionNode input security gate is final | Override attempt raises TypeError |
| TC-07 | FunctionNode output security gate is final | Override attempt raises TypeError |
| TC-08 | required_trust_level is enforced through node(state) | Denial happens before execute |
| TC-09 | PreProcessNode and MainNode extra S-2 hooks | MCP abuse requests blocked; PII in normalized_query masked before retrieval |
| TC-10 | PostProcessNode extra S-3 hook | Credential shapes redacted, placeholders preserved, surviving credential blocked |
| TC-11 | Domain tracing | Each execute path emits a domain event |

## Proof-of-Boundary Tests

| PB-ID | Boundary | Expected Result |
|-------|----------|-----------------|
| PB-1 | BaseNode to trace emitter | Domain and lifecycle trace evidence exists |
| PB-2 | State serialization | Only msgpack-safe annotated fields |
| PB-3 | Template to external service | Injected retriever and LLM doubles exercise the boundary |
| PB-4 | Import isolation | No Level 0 imports |
| PB-5 | Checkpoint safety, conditional | Auto-waived — checkpointing disabled |
| PB-6 | Invoke order | S-1 → node_start → S-2 → execute → S-3 → node_complete |
| PB-7 | HITL interrupt propagation, conditional | Auto-waived — non-HITL |

PB-1 through PB-4 and PB-6 are mandatory. PB-5 applies only when memory or HITL enables
checkpointing and the framework provides both ingress hooks. PB-7 applies only when
config/config.yaml sets hitl.enabled to true.

## Business Logic Tests

| TC-ID | Test | Input | Expected Result |
|-------|------|-------|-----------------|
| BL-01 | Intent extraction | setup, auth, schema, and best-practice questions | Correct intent_class and server fields |
| BL-02 | HTML and script sanitization | unsafe markup | Markup removed, or markup-only input rejected |
| BL-03 | Input bounds | query over the limit, and a two-word query | Actionable input guidance, not a failed invocation |
| BL-04 | Retrieval success | injected retriever returns chunks | JSON chunks and count |
| BL-05 | Retrieval filters | known server and transport | intent, server, transport, and category become filters; `unknown` is omitted |
| BL-06 | Error-intent short-circuit | query mixing setup wording with an error | intent_class is error_troubleshoot |
| BL-07 | Error diagnosis | 401, connection refused, and JSON-RPC parse errors | Matching catalogue pattern, cause, and fix |
| BL-08 | Response synthesis | chunks plus injected LLM | Answer, deduplicated citations, snippets, confidence, canonical output |
| BL-09 | Provider failure | LLM raises | Evidence-only answer carrying retrieved chunks and any diagnosis |
| BL-10 | Snippet extraction | fenced blocks with and without a language tag | Normalized language labels |
| BL-11 | Confidence tiers | chunk counts and scores | High, medium, or low per policy |
| BL-12 | Output safety | generated credential alongside a documented placeholder | Credential redacted, placeholder preserved, formatted_output kept in sync |
| BL-13 | Azure wiring | Azure secrets present | Request-scoped client built from ctx.secrets, no process-global client |
| BL-14 | Missing secrets | no Azure secrets bound | Terminal evidence-only response with visible Azure-unavailable guidance |
| BL-15 | Bearer mapping | external, runner, wrong, and absent bearer | VERIFIED_EXTERNAL, INTERNAL, HTTP 401, and ANONYMOUS respectively |
| BL-16 | Retrieval outage | retriever raises | Safe guidance, backend detail not disclosed, invocation still succeeds |
| BL-17 | Marketplace shaping | same state with and without conversation_history | Marketplace receives the answer text, the API keeps canonical JSON |

## Test Execution Summary

- Execution date: 2026-09-30
- Framework: agenticstar-agentcore 1.0.3 wheel (not a CI import shim)
- Full suite: 87 passed, 0 failed, 3 skipped
- Local gates: manifest schema, trust level, audit trace, dependency pinning, stub tests,
  credential scan, OSS license, and Cat consistency all PASS

The three expected skips are PB-5 (checkpointing disabled) and both PB-7 cases (non-HITL).
