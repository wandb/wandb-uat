#!/usr/bin/env python

import tempfile

import numpy as np
import wandb
from PIL import Image


def main():
    run = wandb.init()
    run_path = run.path

    np_image = np.random.randint(0, high=256, size=(32, 32, 3))
    run.log({"np_image": wandb.Image(np_image)})

    pil_image = Image.fromarray(np_image.astype("uint8"))
    run.log({"pil_image": wandb.Image(pil_image)})

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        pil_image.save(tmp)
        run.log({"path_image": wandb.Image(tmp.name)})

    run.finish()
    check(run_path)


def check(run_path):
    import os

    if os.environ.get("WB_UAT_SKIP_CHECK"):
        return
    api = wandb.Api()
    api_run = api.run(run_path)
    assert api_run.summary["np_image"]["_type"] == "image-file"
    assert api_run.summary["path_image"]["_type"] == "image-file"
    assert api_run.summary["pil_image"]["_type"] == "image-file"
    assert api_run.state == "finished"


if __name__ == "__main__":
    main()
