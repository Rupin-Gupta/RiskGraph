#!/usr/bin/env bash
# Instance role (SPEC §13.2), least privilege: pull the two images, read the two secrets, send
# email from the verified addresses, read the data and artifacts buckets, write container logs,
# plus AmazonSSMManagedInstanceCore for SSM deploys and sessions. No Bedrock: the agents call the
# Gemini API (ADR-012), and its key comes from Secrets Manager.
source "$(dirname "$0")/common.sh"
: "${SES_SENDER:?set SES_SENDER in .env}" "${SES_RECIPIENT:?set SES_RECIPIENT in .env}"

role="$NAME-ec2"
# One entry per verified address (the sender and the recipient are often the same mailbox).
ses_arns="$(printf 'arn:aws:ses:%s:%s:identity/%s\n' "$REGION" "$ACCOUNT" "$SES_SENDER" \
    "$REGION" "$ACCOUNT" "$SES_RECIPIENT" | sort -u | sed 's/.*/"&"/' | paste -sd, -)"
policy="$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
 {"Sid": "PullImages", "Effect": "Allow",
  "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
  "Resource": ["arn:aws:ecr:$REGION:$ACCOUNT:repository/$NAME-api",
               "arn:aws:ecr:$REGION:$ACCOUNT:repository/$NAME-frontend"]},
 {"Sid": "ReadSecrets", "Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
  "Resource": ["arn:aws:secretsmanager:$REGION:$ACCOUNT:secret:$APP_SECRET-*",
               "arn:aws:secretsmanager:$REGION:$ACCOUNT:secret:$TOKEN_SECRET-*"]},
 {"Sid": "SendEscalations", "Effect": "Allow", "Action": ["ses:SendEmail", "ses:SendRawEmail"],
  "Resource": [$ses_arns]},
 {"Sid": "ListData", "Effect": "Allow", "Action": "s3:ListBucket",
  "Resource": ["arn:aws:s3:::$DATA_BUCKET", "arn:aws:s3:::$ARTIFACTS_BUCKET"]},
 {"Sid": "ReadData", "Effect": "Allow", "Action": "s3:GetObject",
  "Resource": ["arn:aws:s3:::$DATA_BUCKET/*", "arn:aws:s3:::$ARTIFACTS_BUCKET/*"]},
 {"Sid": "ContainerLogs", "Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
  "Resource": "arn:aws:logs:$REGION:$ACCOUNT:log-group:$LOG_GROUP:*"}
]}
EOF
)"

plan "IAM role and instance profile $role (trusted by EC2) with AmazonSSMManagedInstanceCore and
this inline policy:
$policy" \
    "\$0 (IAM is free)."

ensure_role "$role" ec2.amazonaws.com
aws iam attach-role-policy --role-name "$role" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name "$role" --policy-name "$role" --policy-document "$policy"
if ! aws iam get-instance-profile --instance-profile-name "$role" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$role" \
        --tags "Key=Project,Value=$NAME" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$role" --role-name "$role"
    echo "created instance profile $role"
fi
echo "role $role is up to date"
