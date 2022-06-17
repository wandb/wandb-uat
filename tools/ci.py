import argparse
import os
import subprocess
import time
from typing import List, Optional, Tuple

import requests  # type: ignore


CIRCLECI_API_TOKEN = "CIRCLECI_TOKEN"


def poll(
    args: argparse.Namespace,
    pipeline_id: Optional[str] = None,
    workflow_ids: Optional[List[str]] = None,
) -> None:
    """
    Poll the CircleCI API for the status of the pipeline.
    borrowed from wandb/wandb

    :param args:
    :param pipeline_id:
    :param workflow_ids:
    :return:
    """
    print(f"Waiting for pipeline to complete (Branch: {args.branch})...")
    if workflow_ids is None:
        workflow_ids = []
    while True:
        num = 0
        done = 0
        if pipeline_id:
            url = (
                f"https://circleci.com/api/v2/pipeline/{pipeline_id}/workflow"
            )
            r = requests.get(url, auth=(args.api_token, ""))
            assert r.status_code == 200, f"Error making api request: {r}"
            d = r.json()
            workflow_ids = [item["id"] for item in d["items"]]
        num = len(workflow_ids)
        for work_id in workflow_ids:
            work_status_url = f"https://circleci.com/api/v2/workflow/{work_id}"
            r = requests.get(work_status_url, auth=(args.api_token, ""))
            assert r.status_code == 200, f"Error making api work request: {r}"
            w = r.json()
            status = w["status"]
            print("Status:", status)
            if status not in ("running", "failing"):
                done += 1
        if num and done == num:
            print("Finished")
            return
        time.sleep(20)


def trigger(args: argparse.Namespace) -> None:
    """
    Trigger a nightly UAT run on CircleCI

    :param args:
    :return:
    """
    url = "https://circleci.com/api/v2/project/gh/wandb/wandb-uat/pipeline"
    default_clouds_set = {"gcp", "aws", "azure"}
    requested_clouds_set = (
        set(args.clouds.split(",")) if args.clouds else default_clouds_set
    )

    # check that all requested shards are valid and that there is at least one
    if not requested_clouds_set.issubset(default_clouds_set):
        raise ValueError(
            f"Requested invalid clouds: {requested_clouds_set}. "
            f"Valid clouds are: {default_clouds_set}"
        )
    # flip the requested clouds to True
    clouds = {
        f"execute_{cloud}": True if cloud in requested_clouds_set else False
        for cloud in default_clouds_set
    }
    if not any(clouds.values()):
        raise ValueError(
            f"Requested no clouds. Valid clouds are: {default_clouds_set}"
        )

    payload = {
        "branch": args.branch,
        "parameters": {
            **{
                "manual": True,
                "slack_notify": args.slack_notify or False,
            },
            **clouds,
        },
    }

    print("Sending to CircleCI:", payload)
    if args.dryrun:
        return
    r = requests.post(url, json=payload, auth=(args.api_token, ""))
    assert r.status_code == 201, "Error making api request"
    d = r.json()
    uuid = d["id"]
    print("CircleCI workflow started:", uuid)
    if args.wait:
        poll(args, pipeline_id=uuid)


def process_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="action", title="action", description="Action to perform"
    )
    parser.add_argument("--api_token", help=argparse.SUPPRESS)
    parser.add_argument("--branch", help="git branch (autodetected)")
    parser.add_argument(
        "--dryrun", action="store_true", help="Don't do anything"
    )

    parse_trigger = subparsers.add_parser("trigger")
    parse_trigger.add_argument(
        "--slack-notify",
        action="store_true",
        help="post notifications to slack",
    )
    parse_trigger.add_argument(
        "--clouds",
        help=(
            "comma-separated list of cloud providers "
            "to run UAT on (gcp,aws,azure)"
        ),
    )
    parse_trigger.add_argument(
        "--wait", action="store_true", help="Wait for finish or error"
    )

    args = parser.parse_args()
    return parser, args


def process_environment(args: argparse.Namespace) -> None:
    api_token = os.environ.get(CIRCLECI_API_TOKEN)
    assert api_token, f"Set environment variable: {CIRCLECI_API_TOKEN}"
    args.api_token = api_token


def process_workspace(args: argparse.Namespace) -> None:
    branch = args.branch
    if not branch:
        code, branch = subprocess.getstatusoutput("git branch --show-current")
        assert code == 0, "failed git command"
        args.branch = branch


def main() -> None:
    parser, args = process_args()
    process_environment(args)
    process_workspace(args)

    if args.action == "trigger":
        trigger(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
