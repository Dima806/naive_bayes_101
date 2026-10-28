"""Typed configuration loaded from ``config/settings.yaml``.

Everything tunable — seeds, smoothing strength, sample sizes, correlation knobs —
lives in the YAML file so notebooks and modules never hard-code these values.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class SmoothingConfig(BaseModel):
    alpha: float = 1.0


class TextConfig(BaseModel):
    cache_dir: str = "data/text"
    max_features: int = 2000
    n_samples: int = 1200


class TabularConfig(BaseModel):
    n_samples: int = 1000


class SyntheticConfig(BaseModel):
    n_samples: int = 2000
    n_features: int = 20
    class_sep: float = 1.5
    correlation: float = 0.0
    sample_sizes: list[int] = [20, 40, 80, 160, 320, 640, 1280]
    correlations: list[float] = [0.0, 0.3, 0.6, 0.9, 0.99]


class Settings(BaseSettings):
    """Project settings, sourced from ``config/settings.yaml``."""

    model_config = SettingsConfigDict(yaml_file=CONFIG_PATH, extra="ignore")

    seed: int = 42
    test_size: float = 0.25
    smoothing: SmoothingConfig = SmoothingConfig()
    text: TextConfig = TextConfig()
    tabular: TabularConfig = TabularConfig()
    synthetic: SyntheticConfig = SyntheticConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # YAML first, then anything passed explicitly to the constructor.
        return (init_settings, YamlConfigSettingsSource(settings_cls))


def load_settings() -> Settings:
    """Return the project settings read from ``config/settings.yaml``."""
    return Settings()


SETTINGS = load_settings()
