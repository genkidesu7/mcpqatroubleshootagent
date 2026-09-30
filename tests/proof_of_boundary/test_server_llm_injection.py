"""Prove standalone Azure provisioning and request-scoped client construction."""

import importlib


class TestServerAzureRuntime:
    def test_server_imports_without_constructing_process_global_llm(self):
        import src.api.server as server

        importlib.reload(server)

        assert server.app is not None
        assert server.agent is not None
        assert server.agent._nodes["main"]._llm is None
        assert server.agent._nodes["post_process"]._llm is None

    def test_exported_azure_values_reach_invocation_secret_provider(self, monkeypatch):
        expected = {
            "AZURE_OPENAI_API_KEY": "dummy-test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.services.ai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
        }
        for key, value in expected.items():
            monkeypatch.setenv(key, value)

        import src.api.server as server

        importlib.reload(server)

        assert {key: server._secrets_provider.require(key) for key in expected} == expected


class TestStandaloneTrustPromotion:
    """BL-15: bearer mapping never elevates an external caller to INTERNAL."""

    def test_external_bearer_never_promotes_to_internal(self):
        import src.api.server as server
        from framework.schemas.trust_level import TrustLevel

        assert (
            server._resolve_standalone_trust(TrustLevel.ANONYMOUS, "Bearer external", "external", "runner")
            is TrustLevel.VERIFIED_EXTERNAL
        )

    def test_runner_bearer_promotes_to_internal(self):
        import src.api.server as server
        from framework.schemas.trust_level import TrustLevel

        assert (
            server._resolve_standalone_trust(TrustLevel.ANONYMOUS, "Bearer runner", "external", "runner")
            is TrustLevel.INTERNAL
        )

    def test_wrong_or_missing_bearer_is_rejected_when_auth_is_enabled(self):
        import pytest
        import src.api.server as server
        from fastapi import HTTPException
        from framework.schemas.trust_level import TrustLevel

        for authorization in ("", "Bearer wrong"):
            with pytest.raises(HTTPException) as exc:
                server._resolve_standalone_trust(TrustLevel.ANONYMOUS, authorization, "external", "runner")
            assert exc.value.status_code == 401

    def test_no_configured_tokens_leaves_caller_anonymous(self):
        import src.api.server as server
        from framework.schemas.trust_level import TrustLevel

        assert server._resolve_standalone_trust(TrustLevel.ANONYMOUS, "", None, None) is TrustLevel.ANONYMOUS
