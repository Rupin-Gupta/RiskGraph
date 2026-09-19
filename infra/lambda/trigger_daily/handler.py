"""Weekday trigger (EventBridge Scheduler -> Lambda, SPEC §13.2).

POSTs /api/runs/daily on the RiskGraph host with the bearer token from Secrets Manager. When the
instance is stopped (make aws-stop) it logs a skip and returns normally, so the error alarm only
fires on real failures. Any HTTP error raises, which is what the alarm watches.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

import boto3

TIMEOUT = 280  # the run refreshes market data first; the Lambda timeout is 300s


def handler(event: Any, context: Any) -> dict[str, Any]:
    ec2 = boto3.client("ec2")
    reservations = ec2.describe_instances(InstanceIds=[os.environ["INSTANCE_ID"]])["Reservations"]
    state = reservations[0]["Instances"][0]["State"]["Name"]
    if state != "running":
        print(json.dumps({"skipped": True, "instance_state": state}))
        return {"skipped": True, "instance_state": state}

    token = boto3.client("secretsmanager").get_secret_value(SecretId=os.environ["TOKEN_SECRET"])[
        "SecretString"
    ]
    request = urllib.request.Request(
        os.environ["SITE_URL"].rstrip("/") + "/api/runs/daily",
        data=b"{}",
        method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = json.loads(response.read())
    summary = {
        "date": body["date"],
        "breaches": len(body["breaches"]),
        "queued": body["queued"],
    }
    print(json.dumps(summary))
    return summary
