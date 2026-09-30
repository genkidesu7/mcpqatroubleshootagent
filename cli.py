"""AGENTIC STAR Marketplace entrypoint for CMN-C1-069 — one-shot Pod process.

Referenced by this repo's Dockerfile as the image `CMD`. Compiles the agent,
provisions its secrets, then hands off to shared.bootstrap.marketplace_app
for the Marketplace lifecycle (identity, input, events, terminal delivery, exit).

`agent_name` and `namespace` must match the values `src/api/server.py` passes to
its own secret provider: both become path segments in the dotenv provider's
lookup (`env/agents/<namespace>/<agent_name>/.env.<env>`), so a mismatch would
make the Marketplace and standalone entrypoints resolve secrets from different
directories.

`namespace` here is the secret-provisioning namespace. It shares its value with
`config/agent.yaml`'s AgentRegistry `namespace:` key but is a distinct concept.
"""

from pathlib import Path

from framework.utils.config_loader import load_agent_config
from shared.bootstrap.marketplace_app import run_agent_marketplace
from src.graph.graph import Graph

# Add config overrides here to set values without touching config/config.yaml.
extend_config: dict[str, object] = {}

if __name__ == "__main__":
    run_agent_marketplace(
        Graph,
        agent_name="McpQaTroubleshootAgent",
        namespace="agent1000",
        config={**load_agent_config(Path(__file__).resolve().parent), **extend_config},
    )
