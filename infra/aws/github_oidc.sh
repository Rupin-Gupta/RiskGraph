#!/usr/bin/env bash
# GitHub Actions OIDC (SPEC §13.2): a provider and a deploy role that only pushes to the two ECR
# repositories and runs the deploy command on this instance. No static AWS keys anywhere.
# Re-run it after the instance exists to add the SSM permission.
source "$(dirname "$0")/common.sh"

role="$NAME-github-deploy"
provider="arn:aws:iam::$ACCOUNT:oidc-provider/token.actions.githubusercontent.com"
sub="repo:$GITHUB_REPO:ref:refs/heads/main"
trust="$(cat <<EOF
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow",
  "Principal": {"Federated": "$provider"},
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {"StringEquals": {
    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
    "token.actions.githubusercontent.com:sub": "$sub"}}}]}
EOF
)"
id="$(instance_id)"
ssm=""
if [[ -n $id ]]; then
    ssm="$(cat <<EOF
,
 {"Sid": "DeployToThisInstance", "Effect": "Allow", "Action": "ssm:SendCommand",
  "Resource": ["arn:aws:ec2:$REGION:$ACCOUNT:instance/$id",
               "arn:aws:ssm:$REGION::document/AWS-RunShellScript"]},
 {"Sid": "ReadCommandResult", "Effect": "Allow",
  "Action": ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations"], "Resource": "*"}
EOF
    )"
fi
policy="$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
 {"Sid": "PushImages", "Effect": "Allow",
  "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
             "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage",
             "ecr:GetDownloadUrlForLayer"],
  "Resource": ["arn:aws:ecr:$REGION:$ACCOUNT:repository/$NAME-api",
               "arn:aws:ecr:$REGION:$ACCOUNT:repository/$NAME-frontend"]}$ssm
]}
EOF
)"

plan "OIDC provider token.actions.githubusercontent.com (audience sts.amazonaws.com).
Role $role, assumable only by $sub, with:
$policy
${id:+Deploys go to instance $id only.}${id:-No instance yet: re-run this after ec2_instance.sh to add the SSM permission.}" \
    "\$0 (IAM and OIDC are free)."

if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$provider" >/dev/null 2>&1; then
    aws iam create-open-id-connect-provider \
        --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com \
        --tags "Key=Project,Value=$NAME" >/dev/null
    echo "created the OIDC provider"
fi
ensure_role "$role" "$trust"
aws iam put-role-policy --role-name "$role" --policy-name "$role" --policy-document "$policy"
arn="$(aws iam get-role --role-name "$role" --query Role.Arn --output text)"
echo "role: $arn"
echo
echo "Set the repository variables GitHub Actions needs (run these yourself):"
echo "  gh variable set AWS_DEPLOY_ROLE_ARN --body '$arn' --repo $GITHUB_REPO"
echo "  gh variable set AWS_REGION --body '$REGION' --repo $GITHUB_REPO"
echo "  gh variable set EC2_INSTANCE_ID --body '${id:-<instance id>}' --repo $GITHUB_REPO"
