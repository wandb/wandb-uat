import argparse
import os
import subprocess
import time

import requests


CIRCLECI_API_TOKEN = "CIRCLECI_TOKEN"


def poll(args, pipeline_id=None, workflow_ids=None):
    """
    Poll the CircleCI API for the status of the pipeline.
    borrowed from wandb/wandb

    :param args:
    :param pipeline_id:
    :param workflow_ids:
    :return:
    """
    print(f"Waiting for pipeline to complete (Branch: {args.branch})...")
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
            # print("STATUS", work_status_url)
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


def trigger(args):
    """
    Trigger a nightly UAT run on CircleCI

    :param args:
    :return:
    """
    url = "https://circleci.com/api/v2/project/gh/wandb/wandb-uat/pipeline"
    default_shards = "cpu,gpu,tpu,local"

    default_shards_set = set(default_shards.split(","))
    requested_shards_set = (
        set(args.shards.split(",")) if args.shards else default_shards_set
    )

    # check that all requested shards are valid and that there is at least one
    if not requested_shards_set.issubset(default_shards_set):
        raise ValueError(
            f"Requested invalid shards: {requested_shards_set}. "
            f"Valid shards are: {default_shards_set}"
        )
    # flip the requested shards to True
    shards = {
        f"manual_nightly_execute_shard_{shard}": True
        for shard in requested_shards_set
    }

    payload = {
        "branch": args.branch,
        "parameters": {
            **{
                "manual": True,
                "manual_nightly": True,
                "manual_nightly_slack_notify": args.slack_notify or False,
            },
            **shards,
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


def process_args():
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


def process_environment(args):
    api_token = os.environ.get(CIRCLECI_API_TOKEN)
    assert api_token, f"Set environment variable: {CIRCLECI_API_TOKEN}"
    args.api_token = api_token


def process_workspace(args):
    branch = args.branch
    if not branch:
        code, branch = subprocess.getstatusoutput("git branch --show-current")
        assert code == 0, "failed git command"
        args.branch = branch


def main():
    parser, args = process_args()
    process_environment(args)
    process_workspace(args)

    if args.action == "trigger":
        trigger(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
