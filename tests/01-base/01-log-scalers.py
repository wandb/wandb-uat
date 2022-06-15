#!/usr/bin/env python

import wandb


def main():
    wandb.init()
    project = wandb.run.project
    run_id = wandb.run.id
    wandb.log(dict(m1=1))
    wandb.finish()
    check(project, run_id)


def check(project, run_id):
    import os
    if os.environ.get("WB_UAT_SKIP_CHECK"):
        return
    api = wandb.Api()
    api_run = api.run(f"{project}/{run_id}")
    assert api_run.summary["m1"] == 1


if __name__ == "__main__":
    main()
