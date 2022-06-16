import argparse
import os
import subprocess
import time
from dataclasses import dataclass, fields


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
    vm_image_name: str = "nvidia-gpu-cloud-image-pytorch-20220228"
    # vm_image_name: str = "nvidia-gpu-cloud-image-tensorflow-20220228"
    docker_image_name: str = "nvcr.io/nvidia/pytorch:22.02-py3"
    python_version: str = "3.8"
    git_branch: str = "main"


class CLI:
    def __init__(self, config: Config, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose

        self.print(self.config)

    def print(  # type: ignore
        self,
        *args,
        sep: str = " ",
        end: str = "\n",
        file=None,
    ) -> None:
        if self.verbose:
            print(*args, sep=sep, end=end, file=file)

    @staticmethod
    def update_components() -> None:
        subprocess.run(["gcloud", "--quiet", "components", "update"])

    def create_vm(self) -> int:
        """
        Create the VM

        - The first command creates a VM similar to the one
          the user can get from the GCP marketplace.
          - There is apparently no way to "interact" with the
            GCP marketplace directly.
        - The VMI explicitly asks to install GPU drivers on the first boot,
          so the second command does it.

        :return:
        """
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
            f"projects/{self.config.project}/global/images/"
            f"{self.config.vm_image_name}",
            "--boot-disk-size",
            self.config.disk_size,
            "--boot-disk-type",
            self.config.disk_type,
            "--accelerator",
            f"type={self.config.accelerator_type},"
            f"count={self.config.accelerator_count}",
        ]
        self.print(" ".join(cmd))
        subprocess.run(cmd)

        # Agree to NVIDIA's prompt and install the GPU driver
        for _ in range(6):
            p = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "ssh",
                    self.config.instance_name,
                ],
                input=b"Y\r\n",
            )
            if p.returncode == 0:
                self.print("GPU driver installed")
                break
            else:
                # allow some time for the VM to boot
                self.print("Waiting for VM to boot...")
                time.sleep(10)

        return p.returncode

    def run_user_acceptance_tests(self) -> int:
        """
        Run the user acceptance tests:
          - ssh into the VM
          - clone the wandb-uat repo
          - spin up a container from the image that comes with the VMI
            - pass the api key to the container
            - pip install wandb
            - run the tests
        :return:
        """
        cmd = [
            "gcloud",
            "compute",
            "ssh",
            self.config.instance_name,
            "--command",
            "git clone https://github.com/wandb/wandb-uat.git; "
            "docker run --gpus all --rm -v ~/wandb-uat:/workspace "
            f"--env WANDB_API_KEY={os.environ.get('WANDB_API_KEY')} "
            f"{self.config.docker_image_name} "
            '/bin/bash -c "pip install wandb[media] && ./bin/test.sh"',
        ]
        self.print(" ".join(cmd))
        p = subprocess.run(cmd)
        return p.returncode

    def delete_vm(self) -> int:
        """
        Delete the VM
        :return:
        """
        p = subprocess.run(
            [
                "gcloud",
                "compute",
                "instances",
                "delete",
                self.config.instance_name,
                "--quiet",
            ]
        )
        return p.returncode


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
