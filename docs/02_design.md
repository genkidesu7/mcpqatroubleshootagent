# Template Design Specification

## Position in AgentCore Architecture

- **Agent Class**: McpQaTroubleshootAgent
- **L1 Base**: AgentBaseGraph
- **Category**: Cat 1
- **Generation Mode**: llm
- **Three-Layer Separation**:
  - State: flat TypedDict fields containing only msgpack-safe data.
  - Node: FunctionNode subclasses override execute and optional extra security hooks.
  - Graph: register_nodes composes the fixed framework pipeline.

## Architecture Overview

| Node | Responsibility | Input State | Output State | Inherits |
|------|----------------|-------------|--------------|----------|
| initialize | Framework invocation metadata | framework-managed | framework-managed | InitializeNode |
| pre_process | Sanitize the query and extract MCP intent, server, transport, and error text | user_input or raw_query | normalized_query and intent fields | FunctionNode |
| main | Retrieve MCP documentation and match the reported error to the failure catalogue | normalized query and intents | retrieved_chunks, retrieval_count, error_diagnosis | FunctionNode |
| post_process | Synthesize or safely degrade, extract snippets and citations, package canonical output | retrieved chunks and diagnosis | answer, citations, code_snippets, confidence, formatted_output | FunctionNode |
| finalize | Framework response metadata | framework-managed | framework-managed | FinalizeNode |

### Data Flow

    START → initialize → pre_process → main → post_process → finalize → END
                                      ↘ retry → pre_process

Graph registers retrieval and request-scoped synthesis separately:

    self._nodes["main"] = MainNode()
    self._nodes["post_process"] = PostProcessNode(
        llm_timeout_s=config["timeout_s"],
        llm_max_retry=config["max_retry"],
    )

Production creates the Azure client from invocation-scoped secrets inside PostProcessNode.
Constructor injection is a test seam only; clients are never written to State or checkpoint surfaces.

### Intent and Diagnosis

Intent classification scores five vocabularies. A query carrying an error short-circuits to
`error_troubleshoot` regardless of other wording, because misrouting a troubleshooting question
is the highest-cost misclassification. An unmatched query defaults to `schema_protocol`, which
routes to specification content rather than a possibly irrelevant setup guide.

Well-known servers (filesystem, postgres, github, and others) resolve transport and category
through a direct lookup; otherwise both are keyword-scored independently and fall back to
`unknown`, in which case they are omitted from the retrieval filters rather than narrowing the
search incorrectly.

Error diagnosis matches the extracted error text against a catalogue of known MCP failure
patterns covering authentication, transport, and protocol faults. The catalogue is deterministic,
so a troubleshooting question still receives an actionable cause and fix when synthesis degrades.

## State Definition

| Field group | Types | Purpose |
|-------------|-------|---------|
| raw_query, normalized_query, query_id | str | Safe query processing and correlation |
| intent_class | str | Question class driving retrieval boost |
| intent_server_name, intent_sdk_client, intent_auth_type | str or None | Retrieval filters |
| intent_error_message | str or None | Error text handed to the diagnosis catalogue |
| server_transport, server_category | str | MCP server shape, `unknown` when undetermined |
| retrieved_chunks | str | JSON-serialized documentation chunk list |
| retrieval_count | int | Number of chunks retrieved |
| error_diagnosis | str | JSON-serialized diagnosis, empty object when not applicable |
| answer, citations, code_snippets, confidence | str | Domain response |
| synthesis_mode | str | `azure_openai` or `evidence_only` |
| formatted_output | str | Canonical JSON response exposed by AgentBaseGraph |
| error_message, recoverable_error_code | str or None | Short-circuit and degradation control |
| input_error_message, input_error_guidance | str or None, list[str] | User-facing input guidance |

**State Constraints:**

- Flat TypedDict only; collections contain primitive or JSON-safe values.
- No API keys, bearer tokens, credentials, retrievers, LLM clients, or Pydantic objects.
- InvocationContext is reconstructed with InvocationContext.from_state(state) inside a node if needed.
- External clients are constructor dependencies or invocation configuration, never persisted fields.

## Framework Utilization

- [x] InvocationContext and caller trust at the graph boundary.
- [x] Framework S-1 required_trust_level on every domain node.
- [x] Framework final S-2/S-3 gates remain non-overridden.
- [x] PreProcessNode extends S-2 through _extra_security_gate_input to block MCP abuse requests.
- [x] MainNode extends S-2 through _extra_security_gate_input for normalized-query PII masking.
- [x] PostProcessNode extends S-3 through _extra_security_gate_output for generated output checks.
- [x] Every execute method emits a domain trace event on success, failure, or short circuit.
- [x] AgentRegistry and standalone server both load config/config.yaml separately from agent.yaml.
- [x] Marketplace and standalone entrypoints bind secrets per invocation and degrade safely when Azure is unavailable.

## Security and Runtime Decisions

- Minimum caller trust: VERIFIED_EXTERNAL.
- External bearer credentials map only to VERIFIED_EXTERNAL.
- A separately configured STG runner token is required for INTERNAL.
- Bearers are compared in constant time and invalid configured credentials return HTTP 401.
- memory_enabled and hitl.enabled are false, so no checkpointer is created.
- A missing retriever returns zero chunks; no MCP methods, fields, or config keys are invented.
- S-2 blocks auth-bypass and unsafe tool-call-injection requests with bounded quantifiers only,
  so a hostile input cannot force quadratic-time backtracking.
- S-3 redacts credential shapes inside the node that produces the content, so the framework
  credential scan sees clean output and the invocation ends in a redacted success. A downstream
  masking node would never be reached, because the producing node's own gate raises first.
- Documented placeholders such as `<your_api_key>`, `${MCP_TOKEN}`, and `process.env.MCP_TOKEN`
  are preserved. Redacting them would destroy the instructional value of a configuration example,
  which is the capability's purpose.
- Azure client creation is lazy, so missing provider configuration does not break startup. At
  invocation time, provider failure emits progress and returns `synthesis_mode=evidence_only`
  with retrieved documentation and any matched diagnosis, and no fabricated facts.

## EU AI Act Art.13 Design-Time Evidence

The proposal declares this developer documentation capability outside Annex III scope.
The README documents intended purpose and limitations, and response metadata exposes
no-results, synthesis mode, and confidence information for user transparency.

## Import Isolation Confirmation

- [x] Source imports framework and shared layers, never the Level 0 platform SDK.
- [x] Agent-local imports use the src package path.

## Design Decision Record

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| L1 base | AgentBaseGraph / AutonomousBaseGraph | AgentBaseGraph | Fixed query, retrieval, and response stages need no autonomous loop |
| Composition | Standalone / subgraph / remote | Standalone | One independently deployable technical capability |
| HITL | Enabled / disabled | Disabled | The capability answers questions and does not authorize decisions |
| Memory | Enabled / disabled | Disabled | Each MCP question is self-contained |
| Retrieval backend | Injected retriever / embedded client | Injected retriever | The deployment owns the index; the template stays backend-agnostic and returns no evidence when unconfigured |
| Credential handling in output | Redact and preserve placeholders / block on match | Redact | Blocking would fail any answer quoting a configuration example, which is the core value |
| Error diagnosis | Deterministic catalogue / LLM-only | Catalogue, fed into the prompt | Troubleshooting still yields an actionable fix when synthesis is unavailable |
