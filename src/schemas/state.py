"""Flat, msgpack-safe state for CMN-C1-069."""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Agent state for McpQaTroubleshootAgent.

    Add only fields specific to this agent below.
    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    """

    # --- Input ---
    raw_query: str
    top_k: int
    score_threshold: float
    response_budget: int
    max_input_length: int

    # --- Pre-process outputs ---
    query_id: str
    normalized_query: str
    intent_class: str  # setup_config | authentication | error_troubleshoot | schema_protocol | best_practice
    intent_server_name: str | None
    intent_error_message: str | None
    intent_sdk_client: str | None
    intent_auth_type: str | None
    server_transport: str  # stdio | http | sse | unknown
    server_category: str  # filesystem | database | api | custom | unknown

    # --- Main (retrieval) outputs ---
    retrieved_chunks: str  # JSON string: list of chunk dicts
    retrieval_count: int
    error_diagnosis: str  # JSON string: {} or {pattern_id, error_type, likely_cause, suggested_fix}

    # --- Post-process outputs ---
    answer: str
    citations: str  # JSON string: list of citation dicts
    code_snippets: str  # JSON string: list of snippet dicts
    confidence: str  # "high" | "medium" | "low"
    synthesis_mode: str  # "azure_openai" | "evidence_only"
    formatted_output: str

    # --- Error ---
    error_message: str | None
    recoverable_error_code: str | None
    input_error_message: str | None
    input_error_guidance: list[str]
