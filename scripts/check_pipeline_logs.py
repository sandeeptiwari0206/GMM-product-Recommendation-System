"""
scripts/check_pipeline_logs.py
==============================
Fetch logs and status for the latest pipeline execution.

Usage
-----
    python scripts/check_pipeline_logs.py
    python scripts/check_pipeline_logs.py --execution-arn <ARN>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, get_sagemaker_config

import boto3


def get_latest_execution_arn(pipeline_name: str, region: str) -> str:
    client = boto3.client("sagemaker", region_name=region)
    resp   = client.list_pipeline_executions(
        PipelineName=pipeline_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    execs = resp.get("PipelineExecutionSummaries", [])
    if not execs:
        raise RuntimeError(f"No executions found for pipeline: {pipeline_name}")
    return execs[0]["PipelineExecutionArn"]


def get_step_statuses(execution_arn: str, region: str):
    client = boto3.client("sagemaker", region_name=region)
    resp   = client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)
    return resp.get("PipelineExecutionSteps", [])


def get_job_name_from_step(step: dict) -> tuple:
    """Return (job_type, job_name) from a pipeline step."""
    meta = step.get("Metadata", {})
    if "ProcessingJob" in meta:
        return "ProcessingJob", meta["ProcessingJob"]["Arn"].split("/")[-1]
    if "TrainingJob" in meta:
        return "TrainingJob", meta["TrainingJob"]["Arn"].split("/")[-1]
    return None, None


def fetch_cloudwatch_logs(log_group: str, log_stream_prefix: str, region: str, max_lines: int = 100):
    """Fetch the last N lines from a CloudWatch log stream."""
    logs   = boto3.client("logs", region_name=region)
    try:
        streams = logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=log_stream_prefix,
            orderBy="LastEventTime",
            descending=True,
            limit=1,
        ).get("logStreams", [])

        if not streams:
            return [f"  No log streams found in {log_group} with prefix {log_stream_prefix}"]

        stream_name = streams[0]["logStreamName"]
        events      = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            limit=max_lines,
            startFromHead=False,
        ).get("events", [])

        return [e["message"] for e in events]

    except logs.exceptions.ResourceNotFoundException:
        return [f"  Log group not found: {log_group}"]
    except Exception as e:
        return [f"  Error fetching logs: {e}"]


def main():
    parser = argparse.ArgumentParser(description="Check pipeline execution logs")
    parser.add_argument("--config",        default="config/config.yaml")
    parser.add_argument("--execution-arn", default=None,
                        help="Specific execution ARN (default: latest)")
    parser.add_argument("--max-lines",     type=int, default=80,
                        help="Max log lines to print per step")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    sm_cfg   = get_sagemaker_config(cfg)
    region   = sm_cfg["region"]
    p_name   = sm_cfg["pipeline_name"]

    # Get execution ARN
    exec_arn = args.execution_arn or get_latest_execution_arn(p_name, region)
    print(f"\nExecution ARN: {exec_arn}\n")

    # Get step statuses
    steps = get_step_statuses(exec_arn, region)

    print("=" * 70)
    print("STEP STATUSES")
    print("=" * 70)
    for s in steps:
        name   = s.get("StepName", "?")
        status = s.get("StepStatus", "?")
        msg    = s.get("FailureReason", "")
        icon   = {"Succeeded": "✅", "Failed": "❌", "Executing": "🔄",
                  "Starting": "⏳", "Stopped": "⛔"}.get(status, "❓")
        print(f"  {icon}  {name:30s}  {status}")
        if msg:
            print(f"       FailureReason: {msg}")

    print()

    # Fetch CloudWatch logs for failed steps
    for s in steps:
        if s.get("StepStatus") != "Failed":
            continue

        step_name       = s.get("StepName", "unknown")
        job_type, j_name = get_job_name_from_step(s)

        print("=" * 70)
        print(f"LOGS FOR FAILED STEP: {step_name}  ({job_type}: {j_name})")
        print("=" * 70)

        if job_type == "ProcessingJob":
            log_group  = "/aws/sagemaker/ProcessingJobs"
            log_prefix = j_name
        elif job_type == "TrainingJob":
            log_group  = "/aws/sagemaker/TrainingJobs"
            log_prefix = j_name
        else:
            print("  Cannot determine log location for this step type.")
            continue

        lines = fetch_cloudwatch_logs(log_group, log_prefix, region, args.max_lines)
        print(f"  Log group : {log_group}")
        print(f"  Job name  : {j_name}")
        print()
        for line in lines:
            print(" ", line.rstrip())

    print("\nDone.")


if __name__ == "__main__":
    main()
