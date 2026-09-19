#!/usr/bin/env bash
# CloudWatch: the container log group, a failed-run metric from the API logs, and alarms on it
# and on trigger Lambda errors, notifying an SNS email subscription (SPEC §13.2).
source "$(dirname "$0")/common.sh"
: "${SES_RECIPIENT:?set SES_RECIPIENT in .env (alarm emails go there)}"

plan "Log group $LOG_GROUP (14-day retention; the awslogs driver writes here).
Metric filter: API log lines 'investigation_failed' or 'dispatch_failed' -> RiskGraph/FailedRuns.
SNS topic $NAME-alarms with an email subscription to SES_RECIPIENT (confirm the email AWS sends).
Alarms: $NAME-failed-runs (FailedRuns >= 1 in a day) and $NAME-trigger-errors (Lambda
$FUNCTION Errors >= 1 in a day), both notifying the topic." \
    "Logs ingestion ~\$0.67/GB (expect well under 0.1 GB/month); 2 alarms at \$0.10 each; SNS email free for 1,000/month: about \$0.30/month."

aws logs create-log-group --log-group-name "$LOG_GROUP" --tags "Project=$NAME" 2>/dev/null ||
    echo "exists: $LOG_GROUP"
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14
aws logs put-metric-filter --log-group-name "$LOG_GROUP" --filter-name "$NAME-failed-runs" \
    --filter-pattern '?investigation_failed ?dispatch_failed' \
    --metric-transformations \
    metricName=FailedRuns,metricNamespace=RiskGraph,metricValue=1,defaultValue=0

topic="$(aws sns create-topic --name "$NAME-alarms" --tags "Key=Project,Value=$NAME" \
    --query TopicArn --output text)"
subscribed="$(aws sns list-subscriptions-by-topic --topic-arn "$topic" \
    --query "Subscriptions[?Endpoint=='$SES_RECIPIENT'] | length(@)" --output text)"
if [[ $subscribed == 0 ]]; then
    aws sns subscribe --topic-arn "$topic" --protocol email \
        --notification-endpoint "$SES_RECIPIENT" >/dev/null
    echo "subscribed $SES_RECIPIENT: confirm the email from AWS Notifications"
fi

aws cloudwatch put-metric-alarm --alarm-name "$NAME-failed-runs" \
    --alarm-description "An investigation or an approved dispatch failed (API logs)." \
    --namespace RiskGraph --metric-name FailedRuns --statistic Sum \
    --period 86400 --evaluation-periods 1 --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold --treat-missing-data notBreaching \
    --alarm-actions "$topic" --tags "Key=Project,Value=$NAME"
aws cloudwatch put-metric-alarm --alarm-name "$NAME-trigger-errors" \
    --alarm-description "The weekday trigger Lambda failed to run /runs/daily." \
    --namespace AWS/Lambda --metric-name Errors --dimensions "Name=FunctionName,Value=$FUNCTION" \
    --statistic Sum --period 86400 --evaluation-periods 1 --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold --treat-missing-data notBreaching \
    --alarm-actions "$topic" --tags "Key=Project,Value=$NAME"
echo "log group, metric filter, topic, and alarms are in place"
