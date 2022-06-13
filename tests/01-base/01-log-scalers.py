#!/usr/bin/env python

import wandb

wandb.init()
project = wandb.run.project
run_id = wandb.run.id
wandb.log(dict(m1=1))
wandb.finish()


#
# Test Checks
#
api = wandb.Api()
api_run = api.run(f"{project}/{run_id}")
assert api_run.summary["m1"] == 1
