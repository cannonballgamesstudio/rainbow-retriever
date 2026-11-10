from __future__ import annotations

import os

from rainbow_retriever.ragd.app import create_app
from rainbow_retriever.ragd.config import RagdConfig, build_backend
from rainbow_retriever.registry import RegistryRO


def build_app(env=None):
    """Construct the ragd ASGI app from environment config.

    This is the missing serving entrypoint: ``create_app`` is a factory that
    needs a backend + registry, and nothing in the package wired it to real
    config. ``build_app`` reads ``RagdConfig.from_env`` (RR_BACKEND,
    RR_EMBED_MODEL, RR_DIMS, RR_REGISTRY_DB, ...), builds the backend,
    opens the registry read-only, and returns the app.

    Run it under uvicorn as a zero-arg factory::

        uvicorn rainbow_retriever.ragd.server:build_app --factory --host 0.0.0.0 --port 8081

    `env` is injectable for tests; it defaults to ``os.environ``.
    """
    env = os.environ if env is None else env
    config = RagdConfig.from_env(env)
    if not config.registry_db:
        raise RuntimeError(
            "RR_REGISTRY_DB must point at the registry sqlite file ragd serves from"
        )
    backend = build_backend(config)
    registry_ro = RegistryRO(config.registry_db)
    return create_app(backend, registry_ro, config)
