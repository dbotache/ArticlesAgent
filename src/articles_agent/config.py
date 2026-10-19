import os
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
# Where per-installation data lives (.env, library/, data/). Defaults to the project folder;
# set ARTICLES_AGENT_HOME to keep it elsewhere, e.g. on a persistent volume when deployed.
PROJECT_ROOT = Path(os.environ["ARTICLES_AGENT_HOME"]).resolve() if os.environ.get("ARTICLES_AGENT_HOME") \
    else CODE_ROOT


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
        if not path.exists():  # a data home without its own config uses the shipped one
            path = CODE_ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data_dir"] = PROJECT_ROOT / cfg["data_dir"]
    return cfg
