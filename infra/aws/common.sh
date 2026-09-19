# Shared settings for the infra/aws scripts (SPEC §13.2). Each script prints what it will create
# and its estimated monthly cost; nothing in AWS changes unless it runs with --apply.
# Prices: AWS Price List API, ap-south-1 on-demand, 2026-09-20.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
envval() { grep -E "^$1=" "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true; }

REGION="${AWS_REGION:-$(envval AWS_REGION)}"
REGION="${REGION:-ap-south-1}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""

NAME=riskgraph
GITHUB_REPO=Rupin-Gupta/RiskGraph
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
SFX="$(printf %s "$ACCOUNT" | shasum -a 256 | cut -c1-8)" # stable suffix that hides the account ID
DATA_BUCKET="$NAME-data-$SFX"
ARTIFACTS_BUCKET="$NAME-artifacts-$SFX"
ECR_REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
LOG_GROUP=/riskgraph/app
APP_SECRET="$NAME/app"
TOKEN_SECRET="$NAME/api-token"
FUNCTION="$NAME-trigger-daily"
SES_SENDER="$(envval SES_SENDER)"
SES_RECIPIENT="$(envval SES_RECIPIENT)"

APPLY=0
for a in "$@"; do [[ $a == --apply ]] && APPLY=1; done

# plan "<what gets created or updated>" "<estimated monthly cost>"
plan() {
    printf '\n== %s (%s) ==\n%s\nEstimated cost: %s\n\n' "$(basename "$0")" "$REGION" "$1" "$2"
    if ((!APPLY)); then
        echo "Plan only: nothing changed. Re-run with --apply to create or update."
        exit 0
    fi
}

instance_id() {
    aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$NAME" \
        "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text
}

eip() {
    aws ec2 describe-addresses --filters "Name=tag:Name,Values=$NAME" \
        --query 'Addresses[0].PublicIp' --output text
}

# ensure_role <name> <trusted service or policy JSON>: create the role if missing
ensure_role() {
    local trust="$2"
    [[ $trust == \{* ]] || trust="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"$trust\"},\"Action\":\"sts:AssumeRole\"}]}"
    if aws iam get-role --role-name "$1" >/dev/null 2>&1; then
        aws iam update-assume-role-policy --role-name "$1" --policy-document "$trust"
    else
        aws iam create-role --role-name "$1" --assume-role-policy-document "$trust" \
            --tags "Key=Project,Value=$NAME" >/dev/null
        echo "created role $1"
    fi
}
