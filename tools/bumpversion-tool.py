#!/usr/bin/env python

import argparse
import configparser
import sys

from bumpversion.cli import main as bumpversion_main
from pkg_resources import parse_version  # type: ignore
import requests  # type: ignore


parser = argparse.ArgumentParser()
parser.add_argument(
    "--to-dev", action="store_true", help="bump release to dev"
)
parser.add_argument(
    "--from-dev", action="store_true", help="bump dev to release"
)
parser.add_argument("--debug", action="store_true", help="debug")
args = parser.parse_args()


def version_problem(current_version: str) -> None:
    print(f"Unhandled version string: {current_version}")
    sys.exit(1)


def get_latest_sdk_version() -> str:
    url = "https://api.github.com/repos/wandb/client/releases/latest"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to get latest sdk version: {response.status_code}")
        sys.exit(1)
    return str(response.json()["tag_name"][1:])


def bump_version(new_version: str) -> None:
    bump_args = []
    if args.debug:
        bump_args += ["--allow-dirty", "--dry-run", "--verbose"]
    bump_args += ["--new-version", new_version, "patch"]
    print(bump_args)
    bumpversion_main(bump_args)


def main() -> None:
    try:
        config = configparser.ConfigParser()
        config.read(".bumpversion.cfg")
        pinned_sdk_version = config["bumpversion"]["pinned_sdk_version"]
        latest_sdk_version = get_latest_sdk_version()

        if parse_version(latest_sdk_version) > parse_version(
            pinned_sdk_version
        ):
            print(f"Latest sdk version is {latest_sdk_version}")
            print(f"Pinned sdk version is {pinned_sdk_version}")
            print("Updating pinned sdk version")
            bump_version(latest_sdk_version)
        else:
            print(
                f"Pinned sdk version {pinned_sdk_version} the latest release."
            )
        sys.exit(0)
    except Exception as e:
        print(f"Failed to update pinned sdk version: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
