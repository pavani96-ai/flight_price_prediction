from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE_PATH = REPO_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = REPO_ROOT / "params" / "params.yaml"
SCHEMA_FILE_PATH = REPO_ROOT / "schema" / "schema.yaml"