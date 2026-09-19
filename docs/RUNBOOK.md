# Runbook

Operational steps for the deployed demo. Every `infra/aws/*.sh` script prints its plan and cost and does nothing without `--apply`.

## Stop everything to save credits

```bash
make aws-stop            # stops the EC2 instance; waits until it is stopped
```

What still bills while it is stopped, about **$6.40/month**:

| Resource | Cost | How to remove it |
|---|---|---|
| Elastic IP | $3.65/month | `aws ec2 release-address` (the site URL and the certificate then change) |
| 30 GB gp3 root volume | $2.74/month | only by terminating the instance |
| Secrets Manager, 2 secrets | $0.80/month | `aws secretsmanager delete-secret` |
| ECR images, S3 data, CloudWatch logs | a few cents | lifecycle policies already cap ECR at 5 images |

The weekday trigger keeps firing while the instance is stopped, sees it is not running, and returns a skip, so no alarm fires.

To stop paying for everything, terminate the instance, release the Elastic IP, and delete the two secrets. Keep the S3 buckets: they hold the DVC data.

## Start it again

```bash
make aws-start           # starts the instance and prints the URL
```

The containers come back on their own (`restart: unless-stopped`). The certificate renews at boot (`@reboot` in `/etc/cron.d/riskgraph-cert`); if the instance was off for more than about six days, give it a minute before the site loads cleanly.

## First deployment, in order

1. `make aws-plan` — read every plan and cost.
2. `infra/aws/s3_buckets.sh --apply`, then push the data: `make data` (or `uvx --from 'dvc[s3]' dvc push -r s3`).
3. `infra/aws/ecr_repos.sh --apply`, `cloudwatch.sh --apply`, `security_group.sh --apply`, `elastic_ip.sh --apply`, `iam_instance_role.sh --apply`.
4. `infra/aws/secrets.sh --apply` (creates the names), fill the values in the console or with `infra/aws/secrets.sh --apply --load-from-env`.
5. `infra/aws/github_oidc.sh --apply`, set the repository variables it prints, and push to `main` so CI builds and pushes the images.
6. `infra/aws/ec2_instance.sh --apply`. Bootstrap takes about ten minutes; watch it with
   `aws ssm start-session --target <id>` then `sudo tail -f /var/log/riskgraph-bootstrap.log`.
7. `infra/aws/github_oidc.sh --apply` again (adds the SSM permission for the new instance) and set `EC2_INSTANCE_ID`.
8. `infra/aws/lambda_trigger.sh --apply`, then test it:
   `aws lambda invoke --function-name riskgraph-trigger-daily /tmp/out.json && cat /tmp/out.json`.
9. `infra/aws/scheduler.sh --apply` (disabled), and `--apply --enable` only when you want the weekday schedule on.

## Daily use

- Open `https://<elastic-ip>` and sign in with the basic-auth credentials.
- Trigger a run by hand:
  `curl -X POST https://<ip>/api/runs/daily -H "authorization: Bearer $API_TOKEN" -H 'content-type: application/json' -d '{"date":"2022-08-16"}'`
- Approve or reject on the incident page. Approval sends the SES email and writes the audit copy under `data/runs/<date>/escalations/`.

## When something breaks

| Symptom | Where to look |
|---|---|
| Site does not load | `aws ssm start-session --target <id>`, then `sudo docker compose -f /opt/riskgraph/docker-compose.yml -f /opt/riskgraph/docker-compose.prod.yml ps` |
| Certificate expired | `sudo /opt/riskgraph/infra/aws/cert.sh` |
| Incident stuck at `failed` | its `reason` column; a 429 means the Gemini daily quota is spent (resets midnight Pacific). Re-post the date to retry |
| Deploy failed in CI | the SSM output printed in the Actions log; the instance must be running |
| Alarm email | CloudWatch log group `/riskgraph/app`, filter `investigation_failed` or `dispatch_failed` |
