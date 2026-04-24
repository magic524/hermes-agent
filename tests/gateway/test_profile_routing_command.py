"""Tests for gateway per-chat profile routing."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.profile_routing import ChatProfileRouter
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/profile", platform=Platform.WEIXIN, chat_id="wx-chat", user_id="u1"):
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="dm",
        user_id=user_id,
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._profile_router = MagicMock()
    runner._profile_router.gateway_profile_name = "ram"
    runner._profile_router.get_routed_profile_name.return_value = "ram"
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    return runner


class TestProfileRoutingCommand:
    @pytest.mark.asyncio
    async def test_profile_command_reports_current_chat_route(self, monkeypatch):
        runner = _make_runner()
        event = _make_event("/profile")

        monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "ram")
        monkeypatch.setattr(
            "hermes_constants.display_hermes_home",
            lambda: "~/.hermes/profiles/ram",
        )

        result = await runner._handle_profile_command(event)

        assert "**This chat routes to:** `ram`" in result

    @pytest.mark.asyncio
    async def test_profile_command_switches_chat_route(self, monkeypatch):
        runner = _make_runner()
        event = _make_event("/profile rem")

        monkeypatch.setattr(
            gateway_run,
            "profile_dir_from_name",
            lambda name: gateway_run.Path(f"/tmp/{name}"),
        )

        result = await runner._handle_profile_command(event)

        runner._profile_router.set_routed_profile_name.assert_called_once_with(event.source, "rem")
        assert event.source.route_tag == "rem"
        assert "routes to profile `rem`" in result

    @pytest.mark.asyncio
    async def test_profile_command_reset_clears_chat_route(self, monkeypatch):
        runner = _make_runner()
        runner._profile_router.get_routed_profile_name.return_value = "rem"
        event = _make_event("/profile default")

        result = await runner._handle_profile_command(event)

        runner._profile_router.set_routed_profile_name.assert_called_once_with(event.source, "ram")
        assert event.source.route_tag == "ram"
        assert "routes to the gateway profile: `ram`" in result

    def test_load_gateway_config_follows_runtime_hermes_home(self, tmp_path, monkeypatch):
        root = tmp_path / ".hermes"
        ram_home = root / "profiles" / "ram"
        rem_home = root / "profiles" / "rem"
        ram_home.mkdir(parents=True)
        rem_home.mkdir(parents=True)

        (ram_home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"default": "gpt-4.1"}}),
            encoding="utf-8",
        )
        (rem_home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"default": "gpt-5.4"}}),
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(ram_home))
        assert gateway_run._load_gateway_config()["model"]["default"] == "gpt-4.1"

        monkeypatch.setenv("HERMES_HOME", str(rem_home))
        assert gateway_run._load_gateway_config()["model"]["default"] == "gpt-5.4"

    def test_chat_profile_router_persists_routes(self, tmp_path, monkeypatch):
        gateway_home = tmp_path / ".hermes" / "profiles" / "ram"
        gateway_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(gateway_home))

        router = ChatProfileRouter(gateway_home)
        source = SessionSource(
            platform=Platform.WEIXIN,
            chat_id="wx-chat",
            chat_type="dm",
        )

        router.set_routed_profile_name(source, "rem")

        reloaded = ChatProfileRouter(gateway_home)
        assert reloaded.get_routed_profile_name(source) == "rem"
