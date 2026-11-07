import yaml


def load_config(path):
    """config.yaml holds `language`, `data`, `preprocess`, `model`, `train` sections."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def split_config(cfg):
    return cfg["preprocess"], cfg["model"], cfg["train"]
