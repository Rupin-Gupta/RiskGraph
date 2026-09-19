#!/usr/bin/env bash
# Deploy on the host (root): update the repo, render .env from Secrets Manager, pull the images
# for a tag (default latest), and restart what changed. CI runs it through SSM Run Command.
set -euo pipefail

main() { # a function, so bash has parsed all of it before git pull can change this file
    . /etc/riskgraph/settings
    cd /opt/riskgraph
    git pull --ff-only -q
    local tag="${1:-latest}"
    secret() {
        aws secretsmanager get-secret-value --region "$REGION" --secret-id "$1" \
            --query SecretString --output text
    }
    umask 077
    {
        secret riskgraph/app | jq -r 'to_entries[]
            | select(.value != "" and (.key | startswith("BASIC_AUTH") | not))
            | "\(.key)=\(.value)"'
        echo "API_TOKEN=$(secret riskgraph/api-token)"
        printf '%s\n' "AWS_REGION=$REGION" "ECR_REGISTRY=$ECR_REGISTRY" "IMAGE_TAG=$tag" \
            "SES_SENDER=$SES_SENDER" "SES_RECIPIENT=$SES_RECIPIENT"
    } >.env.new
    mv .env.new .env
    aws ecr get-login-password --region "$REGION" |
        docker login -u AWS --password-stdin "$ECR_REGISTRY" >/dev/null
    local compose=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)
    "${compose[@]}" pull -q
    "${compose[@]}" up -d --remove-orphans --wait
    docker image prune -f >/dev/null
    echo "deployed $tag"
}

main "$@"
