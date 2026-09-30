"""Standalone FastAPI adapter for CMN-C1-069."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.secrets import factory as secrets_factory
from shared.secrets.chained_provider import ChainedSecretProvider
from shared.secrets.env_provider import EnvProvider
from src.graph.graph import Graph

AGENT_NAME = "McpQaTroubleshootAgent"
app = FastAPI(title=AGENT_NAME)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config: dict[str, Any] = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}

_namespace = "agent1000"
_secrets_provider = ChainedSecretProvider(
    EnvProvider(namespace=_namespace, agent_name=AGENT_NAME),
    secrets_factory(namespace=_namespace, agent_name=AGENT_NAME),
)

agent = Graph(config=_config)
_hitl_enabled = bool(agent.config.get("hitl", {}).get("enabled", False))
_needs_checkpointer = bool(agent.config.get("memory_enabled") or _hitl_enabled)
agent.compile(checkpointer=MemorySaver() if _needs_checkpointer else None)
agent.provision_secrets(_secrets_provider)


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""


def _bearer_matches(supplied: str, expected: str) -> bool:
    """Compare a supplied bearer header in constant time."""
    return secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel,
    authorization: str,
    invoke_auth_token: str | None,
    internal_runner_token: str | None,
) -> TrustLevel:
    """Authenticate standalone callers without external-token elevation."""
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust = _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return cast(dict[str, Any], agent.invoke(req.input, ctx=ctx))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": AGENT_NAME}
