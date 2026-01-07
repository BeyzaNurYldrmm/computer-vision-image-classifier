# toml loader

#tomllib → TOML dosyalarını okumak için kullanılan standart kütüphane
import tomllib #!python --version => 3.11
# python < 3.11 için: import tomli as tomllib
from pathlib import Path

"""
"r" girersek  string döner ve tomllib.load() hata verir. Cunku toml parser'i bayt olarak bekler.
"""

def load_config(path):
    path = Path(path)

    with open(path, "rb") as f:# r=> read, b=>binary, "rb"=> dosya ikili okuma modunda
        cfg = tomllib.load(f)

    if "extends" not in cfg:
        return cfg

    parent_path = path.parent / cfg["extends"]
    parent_cfg = load_config(parent_path)

    return deep_merge(parent_cfg, cfg)


def deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base
