"""Run after pip install -e '.[dev]'."""

import argparse
import json
import logging
from pathlib import Path

from quality_intelligence.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--records", type=int)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(
        args.output,
        n=args.records if args.records is not None else config["records"],
        seed=args.seed if args.seed is not None else config["seed"],
        rate=config["defect_rate"],
        batch_size=config["batch_size"],
        fn_cost=config["false_negative_cost"],
        fp_cost=config["false_positive_cost"],
    )
