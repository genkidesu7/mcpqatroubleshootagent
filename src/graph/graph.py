"""Graph definition for CMN-C1-069 (fixed five-node pipeline)."""

import json
from pathlib import Path
from typing import Any, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.utils.config_loader import load_config
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class Graph(AgentBaseGraph):
    """MCP server integration Q&A and troubleshooting graph."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Marketplace constructs Graph() directly. Load the same runtime config
        # used by the standalone HTTP adapter so both entrypoints behave alike.
        runtime_config = config
        if runtime_config is None:
            runtime_config = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}
        super().__init__(config=runtime_config)

    @property
    def name(self) -> str:
        return "McpQaTroubleshootAgent"

    @property
    def state_schema(self) -> type[State]:
        return State

    def _extra_initial_state(self) -> dict[str, Any]:
        """Expose runtime settings to nodes for every entrypoint."""
        return {
            "top_k": int(self.config.get("top_k", 8)),
            "score_threshold": float(self.config.get("score_threshold", 0.6)),
            "response_budget": int(self.config.get("llm_max_tokens", 1500)),
            "max_input_length": int(self.config.get("max_input_length", 1000)),
        }

    def get_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Keep API metadata intact and expose the MCP answer to Marketplace."""
        envelope = cast(dict[str, Any], super().get_output(state))
        context = state.get("input_context")
        is_marketplace = isinstance(context, dict) and "conversation_history" in context
        if not is_marketplace:
            return envelope

        if state.get("input_error_message"):
            lines = [
                "The MCP question could not be processed.",
                "",
                f"Reason: {state['input_error_message']}",
            ]
            guidance = state.get("input_error_guidance")
            if isinstance(guidance, list) and guidance:
                lines.extend(["", "How to continue:"])
                lines.extend(f"- {item}" for item in guidance)
            envelope["output"] = "\n".join(lines)
            return envelope

        response = self._parse_response(envelope.get("output"))
        if response is not None:
            answer = str(response.get("answer", "")).strip()
            if answer:
                envelope["output"] = answer
        return envelope

    @staticmethod
    def _parse_response(output: Any) -> dict[str, Any] | None:
        if isinstance(output, dict):
            return output
        if not isinstance(output, str):
            return None
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def register_nodes(self) -> None:
        super().register_nodes()

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode()
        self._nodes["post_process"] = PostProcessNode(
            llm_timeout_s=float(self.config.get("timeout_s", 30.0)),
            llm_max_retry=int(self.config.get("max_retry", 3)),
        )
