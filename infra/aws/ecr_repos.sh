#!/usr/bin/env bash
# ECR repositories for the API and dashboard images (SPEC §13.2).
source "$(dirname "$0")/common.sh"

plan "ECR repositories $NAME-api and $NAME-frontend: scan on push, keep the 5 newest images." \
    "\$0.10/GB-month of stored layers: about \$0.10-0.30/month (the API image is ~1 GB compressed, mostly torch; images share layers)."

for r in api frontend; do
    repo="$NAME-$r"
    if aws ecr describe-repositories --repository-names "$repo" >/dev/null 2>&1; then
        echo "exists: $repo"
    else
        aws ecr create-repository --repository-name "$repo" \
            --image-scanning-configuration scanOnPush=true \
            --tags "Key=Project,Value=$NAME" >/dev/null
        echo "created: $repo"
    fi
    aws ecr put-lifecycle-policy --repository-name "$repo" --lifecycle-policy-text \
        '{"rules":[{"rulePriority":1,"description":"keep the 5 newest images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}' \
        >/dev/null
done
