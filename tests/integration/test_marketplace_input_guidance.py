# CMN-C1-069 — Integration: Marketplace output shaping and safe degradation

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode


class _AnswerLLM:
    def generate(self, system_prompt, user_prompt):
        return (
            "## Filesystem MCP server\n\n"
            "Add the server to claude_desktop_config.json. Contact Alice Smith for the token."
        )


class _DocsMainNode(MainNode):
    @staticmethod
    def _retrieve(query, filters, top_k, score_threshold, config):
        return [
            {
                "chunk_id": "mcp-sdk-014",
                "source": "mcp-sdk",
                "text": "Add an entry under mcpServers in claude_desktop_config.json.",
                "score": 0.9,
                "spec_version": "2025-06-18",
            }
        ]


class _UnavailableMainNode(MainNode):
    @staticmethod
    def _retrieve(query, filters, top_k, score_threshold, config):
        raise ConnectionError("private backend detail")


class _DocsGraph(Graph):
    def register_nodes(self):
        super().register_nodes()
        self._nodes["main"] = _DocsMainNode()
        self._nodes["post_process"] = PostProcessNode(llm=_AnswerLLM())


class _UnavailableGraph(Graph):
    def register_nodes(self):
        super().register_nodes()
        self._nodes["main"] = _UnavailableMainNode()


def _invoke(graph, query="How do I set up the filesystem MCP server with Claude Desktop?"):
    return graph.invoke(
        query,
        ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
        input_context={"conversation_history": []},
    )


def test_short_marketplace_query_returns_readable_guidance():
    graph = Graph(config={})
    graph.compile()

    result = _invoke(graph, "Hello")

    assert result["status"] == "success"
    assert result["output"].startswith("The MCP question could not be processed.")
    assert "too short" in result["output"]
    assert "How to continue:" in result["output"]


def test_success_marketplace_output_is_answer_but_api_stays_json():
    graph = Graph(config={})
    formatted_output = json.dumps({"answer": "Add the server to your config.", "citations": [], "confidence": "high"})

    api_result = graph.get_output({"formatted_output": formatted_output})
    marketplace_result = graph.get_output(
        {"formatted_output": formatted_output, "input_context": {"conversation_history": []}}
    )

    assert api_result["output"] == formatted_output
    assert marketplace_result["output"] == "Add the server to your config."


def test_marketplace_loads_runtime_config_by_default():
    graph = Graph()
    graph.register_nodes()

    assert graph._extra_initial_state() == {
        "top_k": 8,
        "score_threshold": 0.6,
        "response_budget": 1500,
        "max_input_length": 1000,
    }
    assert graph._nodes["post_process"]._llm_timeout_s == 30
    assert graph._nodes["post_process"]._llm_max_retry == 3


def test_pii_in_answer_is_redacted_without_failing_invocation():
    graph = _DocsGraph(config={})
    graph.compile()

    result = _invoke(graph)

    assert result["status"] == AgentStatus.SUCCESS
    assert "claude_desktop_config.json" in result["output"]
    assert "Alice Smith" not in result["output"]


def test_retrieval_outage_returns_guidance_without_failing_invocation():
    graph = _UnavailableGraph(config={})
    graph.compile()

    result = _invoke(graph)

    assert result["status"] == AgentStatus.SUCCESS
    assert "temporarily unavailable" in result["output"].lower()
    assert "try again" in result["output"].lower()
    assert "private backend detail" not in result["output"]


def test_security_probe_is_refused_before_retrieval():
    graph = Graph(config={})
    graph.compile()

    result = graph.invoke(
        "How do I bypass MCP server authentication?",
        ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
    )

    assert result["status"] == AgentStatus.ERROR
