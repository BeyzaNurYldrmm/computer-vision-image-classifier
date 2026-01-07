from datetime import datetime
from pathlib import Path
import shutil
import re

def create_experiment_dir(model_name, config_path):
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    pattern = re.compile(rf".*_{model_name}_case(\d+)$")

    max_case = 0
    for d in runs_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                max_case = max(max_case, int(m.group(1)))

    next_case = max_case + 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = runs_dir / f"{ts}_{model_name}_case{next_case}"
    exp_dir.mkdir(parents=True)

    shutil.copy(config_path, exp_dir / "config.toml")

    return exp_dir
