#!/usr/bin/env bash
# EventBridge Scheduler: weekdays at 18:00 IST, invoke the trigger Lambda (SPEC §13.2).
# Created disabled; pass --enable to turn it on.
source "$(dirname "$0")/common.sh"

state=DISABLED
[[ " $* " == *" --enable "* ]] && state=ENABLED
role="$NAME-scheduler"
fn_arn="arn:aws:lambda:$REGION:$ACCOUNT:function:$FUNCTION"

plan "Schedule $NAME-daily: cron(0 18 ? * MON-FRI *) in Asia/Kolkata, flexible window off,
target $FUNCTION, state $state.
Role $role, trusted by scheduler.amazonaws.com, allowed lambda:InvokeFunction on that function only." \
    "\$0 (the first 14 million scheduler invocations a month are free)."

ensure_role "$role" scheduler.amazonaws.com
aws iam put-role-policy --role-name "$role" --policy-name "$role" --policy-document "$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": "$fn_arn"}]}
EOF
)"
role_arn="$(aws iam get-role --role-name "$role" --query Role.Arn --output text)"

args=(--name "$NAME-daily" --schedule-expression "cron(0 18 ? * MON-FRI *)"
    --schedule-expression-timezone Asia/Kolkata --flexible-time-window "Mode=OFF"
    --target "{\"Arn\":\"$fn_arn\",\"RoleArn\":\"$role_arn\",\"Input\":\"{}\"}" --state "$state")
if aws scheduler get-schedule --name "$NAME-daily" >/dev/null 2>&1; then
    aws scheduler update-schedule "${args[@]}" >/dev/null
    echo "updated: $NAME-daily ($state)"
else
    aws scheduler create-schedule "${args[@]}" >/dev/null
    echo "created: $NAME-daily ($state)"
fi
