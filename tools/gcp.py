import argparse
import logging
import os
import subprocess
import sys
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
    test_args: str = "--all"


class CLI:
    def __init__(
        self,
        config: Config,
        verbose: bool = False,
        log_level: int = logging.INFO,
    ) -> None:
        self.config = config
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler("gcp.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        self.print("Initialized CLI")
        self.print(self.config)

    def print(  # type: ignore
        self,
        *args,
        sep: str = " ",
        end: str = "\n",
        file=None,
    ) -> None:
        self.logger.info(sep.join(map(str, args)))
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
        p = subprocess.run(cmd)

        # Agree to NVIDIA's prompt and install the GPU driver.
        # This monster below is here bc the yes command
        # and a gazillion alternatives do not work on circleci.
        # reverse-engineered from /usr/bin/gcp-ngc-login.sh
        cmd = [
            "gcloud",
            "compute",
            "ssh",
            self.config.instance_name,
            "--command",
            "source /etc/nvidia-vmi-version.txt; "
            'REGISTRY="nvcr.io"; NVIDIA_DIR="/var/tmp/nvidia"; '
            "sudo gsutil cp "
            "gs://nvidia-ngc-drivers-us-public/TESLA/shim/NVIDIA-Linux-x86_64-"
            "${NVIDIA_DRIVER_VERSION}-${NVIDIA_GCP_VERSION}-shim.run "
            "${NVIDIA_DIR}; "
            "sudo chmod u+x ${NVIDIA_DIR}/NVIDIA-Linux-x86_64-"
            "${NVIDIA_DRIVER_VERSION}-${NVIDIA_GCP_VERSION}-shim.run; "
            "sudo ${NVIDIA_DIR}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}-"
            "${NVIDIA_GCP_VERSION}-shim.run --no-cc-version-check "
            "--kernel-module-only --silent --dkms; "
            "sudo dkms add nvidia/${NVIDIA_DRIVER_VERSION} || true; "
            "cd /usr/share/doc/NVIDIA_GLX-1.0/samples/; "
            "sudo tar xvjf nvidia-persistenced-init.tar.bz2; "
            "sudo nvidia-persistenced-init/install.sh && "
            "sudo rm -rf nvidia-persistenced-init; ",
        ]
        self.print(cmd)
        for _ in range(6):
            p = subprocess.run(cmd)
            # input=b"Y\r\n"
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
            f"--env WANDB_PROJECT={os.environ.get('WANDB_PROJECT')} "
            f"{self.config.docker_image_name} "
            f'/bin/bash -c "pip install wandb[media] && '
            f'./bin/test.sh {self.config.test_args}"',
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
    exit_code = getattr(cli, command)()
    sys.exit(exit_code)
