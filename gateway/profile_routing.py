"""Helpers for per-chat virtual profile routing in the gateway."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv
from utils import atomic_json_write

if TYPE_CHECKING:
    from .session import SessionSource


logger = logging.getLogger(__name__)


def is_named_profile_home(path: Path) -> bool:
    parts = path.resolve().parts
    return len(parts) >= 2 and parts[-2] == "profiles"


def profile_name_from_home(path: Path) -> str:
    resolved = path.resolve()
    return resolved.name if is_named_profile_home(resolved) else "default"


def list_available_profile_names() -> list[str]:
    from hermes_cli.profiles import list_profiles

    names: list[str] = []
    for info in list_profiles():
        if info.name not in names:
            names.append(info.name)
    return names


def profile_dir_from_name(name: str) -> Path:
    from hermes_cli.profiles import get_profile_dir, validate_profile_name

    normalized = (name or "").strip().lower()
    if normalized in {"", "default", "main", "host"}:
        return get_hermes_home()
    validate_profile_name(normalized)
    return get_profile_dir(normalized)


class ChatProfileRouter:
    """Stores and resolves per-chat profile overrides for the gateway."""

    def __init__(self, gateway_home: Path | None = None):
        self.gateway_home = gateway_home or get_hermes_home()
        self.gateway_profile_name = profile_name_from_home(self.gateway_home)
        self.routes_path = self.gateway_home / "gateway_chat_profile_routes.json"
        self._runtime_lock = threading.RLock()
        self._routes = self._load_routes()

    def _load_routes(self) -> Dict[str, str]:
        try:
            raw = json.loads(self.routes_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("Failed to load chat profile routes from %s: %s", self.routes_path, exc)
            return {}

        if not isinstance(raw, dict):
            return {}

        routes: Dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            normalized = value.strip().lower()
            if normalized:
                routes[key] = normalized
        return routes

    def _save_routes(self) -> None:
        self.routes_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(self.routes_path, self._routes)

    def routing_key_for_source(self, source: "SessionSource") -> str:
        platform = source.platform.value if source.platform else "unknown"
        chat_id = str(source.chat_id or "").strip()
        if chat_id:
            return f"{platform}:{chat_id}"
        return f"{platform}:{source.user_id or source.user_name or 'unknown'}"

    def get_routed_profile_name(self, source: "SessionSource") -> str:
        return self._routes.get(
            self.routing_key_for_source(source),
            self.gateway_profile_name,
        )

    def set_routed_profile_name(self, source: "SessionSource", profile_name: str) -> None:
        key = self.routing_key_for_source(source)
        normalized = (profile_name or "").strip().lower()
        if normalized in {"", "default", self.gateway_profile_name}:
            self._routes.pop(key, None)
        else:
            self._routes[key] = normalized
        self._save_routes()

    @contextmanager
    def runtime_context(self, profile_name: str):
        target_home = profile_dir_from_name(profile_name)
        target_name = profile_name_from_home(target_home)
        if target_name == self.gateway_profile_name:
            yield target_home
            return

        with self._runtime_lock:
            old_env = dict(os.environ)
            try:
                os.environ["HERMES_HOME"] = str(target_home)
                load_hermes_dotenv(
                    hermes_home=target_home,
                    project_env=Path(__file__).resolve().parents[1] / ".env",
                )
                yield target_home
            finally:
                os.environ.clear()
                os.environ.update(old_env)
