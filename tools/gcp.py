import argparse
import os
import subprocess
from dataclasses import dataclass, fields
from typing import Literal, get_args


@dataclass
class Config:
    instance_name: str = "wandb-uat-nvidia-gpu-cloud-pytorch"
    num_nodes: int = 2
    machine_type: str = "n1-custom-14-86016"
    maintenance_policy: str = "TERMINATE"
    disk_size: str = "100GB"
    disk_type: str = "pd-ssd"
    accelerator_type: str = "nvidia-tesla-t4"
    accelerator_count: int = 1
    container_registry: str = "gcr.io"
    gcp_project_id: str = "wandb-client-cicd"
    project: str = "nvidia-ngc-public"
    image_name: str = "nvidia-gpu-cloud-image-pytorch-20220228"
    python_version: str = "3.8"
    git_branch: str = "nightly"


class CLI:
    def __init__(self, config: Config, verbose: bool = False):
        self.config = config
        self.verbose = verbose

        self.print(self.config)

    def print(self, *args, sep=" ", end="\n", file=None):
        if self.verbose:
            print(*args, sep=sep, end=end, file=file)

    @staticmethod
    def update_components():
        subprocess.run(["gcloud", "--quiet", "components", "update"])

    def create_vm(self):
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "create",
            self.config.instance_name,
            "--machine-type",
            self.config.machine_type,
            "--maintenance-policy",
            self.config.maintenance_policy,
            "--image",
            f"projects/{self.config.project}/global/images/{self.config.image_name}",
            "--boot-disk-size",
            self.config.disk_size,
            "--boot-disk-type",
            self.config.disk_type,
            "--accelerator",
            f"type={self.config.accelerator_type},count={self.config.accelerator_count}",
        ]
        self.print(" ".join(cmd))
        subprocess.run(cmd)

        # Agree to NVIDIA's prompt and install the GPU driver
        subprocess.run(
            [
                "gcloud",
                "compute",
                "ssh",
                self.config.instance_name,
                # "--command",
                # "sudo nvidia-smi",
            ],
            input=b"Y\n",
        )

    def run_user_acceptance_tests(self):
        subprocess.run(
            [
                "gcloud",
                "compute",
                "ssh",
                self.config.instance_name,
                "--command",
                # "git clone https://github.com/wandb/wandb-uat.git"
                f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY')};"
                "cd wandb-uat;"
                "./bin/test.sh",
            ]
        )

    def delete_vm(self):
        subprocess.run(
            [
                "gcloud",
                "compute",
                "instances",
                "delete",
                self.config.instance_name,
                "--quiet",
            ]
        )

    # docker pull nvcr.io/nvidia/pytorch:22.05-py3
    # docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:<xx.xx>-py3 <command>


if __name__ == "__main__":
    actions = [
        func
        for func in dir(CLI)
        if callable(getattr(CLI, func)) and not func.startswith("__")
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=actions, help="command to run")
    # add verbose option
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print verbose output",
    )
    for field in fields(Config):
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=field.default,
            help=f"type: {field.type.__name__}; default: {field.default}",
        )

    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    v = arguments.pop("verbose")

    cli = CLI(config=Config(**arguments), verbose=v)
    getattr(cli, command)()