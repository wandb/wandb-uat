#!/usr/bin/env python

import os

import wandb


def main() -> None:
    run = wandb.init()
    run_path = run.path
    run.log(dict(m1=1))
    run.finish()

    if not os.environ.get("WB_UAT_SKIP_CHECK"):
        check(run_path)


def check(run_path: str) -> None:
    api = wandb.Api()
    api_run = api.run(run_path)
    assert api_run.summary["m1"] == 1
    assert api_run.state == "finished"


if __name__ == "__main__":
    main()
