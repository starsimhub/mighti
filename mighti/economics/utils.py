import pandas as pd, yaml
from pathlib import Path

def csv_to_yaml(csv_path, yaml_path=None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    total_budget = float(df["budget_usd"].max()) if "budget_usd" in df else 0
    hrh_cols = [c for c in df.columns if c.endswith("_min")]
    hrh_minutes = {col.replace("_min", ""): float(df[col].sum()) for col in hrh_cols}

    out_dict = dict(
        budget_usd=total_budget,
        hrh_minutes=hrh_minutes,
        rollover=bool(df.get("rollover", [True]).iloc[0]),
        enforce=bool(df.get("enforce", [True]).iloc[0])
    )

    yaml_path = yaml_path or csv_path.with_suffix(".yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(out_dict, f, sort_keys=False)

    return yaml_path