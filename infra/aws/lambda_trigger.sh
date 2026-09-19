#!/usr/bin/env bash
# The weekday trigger Lambda (SPEC §13.2): POSTs /api/runs/daily with the token from Secrets
# Manager, and skips quietly when the instance is stopped.
source "$(dirname "$0")/common.sh"

id="$(instance_id)"
ip="$(eip)"
[[ -n $id && $ip != None ]] || { echo "run ec2_instance.sh --apply first" >&2; exit 1; }
role="$NAME-lambda-trigger"
policy="$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Sid": "ReadToken", "Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
  "Resource": "arn:aws:secretsmanager:$REGION:$ACCOUNT:secret:$TOKEN_SECRET-*"},
 {"Sid": "CheckInstance", "Effect": "Allow", "Action": "ec2:DescribeInstances", "Resource": "*"}
]}
EOF
)"

plan "Lambda $FUNCTION (python3.13, 128 MB, 300 s timeout) from infra/lambda/trigger_daily/,
with SITE_URL=https://$ip, INSTANCE_ID=$id, TOKEN_SECRET=$TOKEN_SECRET.
Role $role: AWSLambdaBasicExecutionRole plus
$policy
Log group /aws/lambda/$FUNCTION with 14-day retention." \
    "About 22 invocations a month at 128 MB: \$0 (well inside the always-free Lambda tier)."

ensure_role "$role" lambda.amazonaws.com
aws iam attach-role-policy --role-name "$role" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam put-role-policy --role-name "$role" --policy-name "$role" --policy-document "$policy"
role_arn="$(aws iam get-role --role-name "$role" --query Role.Arn --output text)"

zip="$(mktemp -d)/trigger.zip"
(cd "$ROOT/infra/lambda/trigger_daily" && zip -q -r "$zip" handler.py)
vars="Variables={SITE_URL=https://$ip,INSTANCE_ID=$id,TOKEN_SECRET=$TOKEN_SECRET}"
if aws lambda get-function --function-name "$FUNCTION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$FUNCTION" --zip-file "fileb://$zip" >/dev/null
    aws lambda wait function-updated --function-name "$FUNCTION"
    aws lambda update-function-configuration --function-name "$FUNCTION" --environment "$vars" \
        --timeout 300 --memory-size 128 >/dev/null
    echo "updated: $FUNCTION"
else
    for attempt in 1 2 3 4 5; do # the new role takes a moment to become assumable
        aws lambda create-function --function-name "$FUNCTION" --runtime python3.13 \
            --handler handler.handler --role "$role_arn" --zip-file "fileb://$zip" \
            --timeout 300 --memory-size 128 --environment "$vars" \
            --tags "Project=$NAME" >/dev/null 2>&1 && break
        echo "waiting for the role to propagate ($attempt)"
        sleep 10
    done
    echo "created: $FUNCTION"
fi
aws logs put-retention-policy --log-group-name "/aws/lambda/$FUNCTION" --retention-in-days 14 2>/dev/null ||
    true # the group appears on the first invocation
echo "test it with: aws lambda invoke --function-name $FUNCTION /tmp/out.json && cat /tmp/out.json"
