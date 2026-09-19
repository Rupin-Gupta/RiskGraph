#!/usr/bin/env bash
# First boot of the RiskGraph host, run as root from EC2 user data (ADR-014). Installs Docker and
# the AWS CLI, records the host settings, pulls the DVC data from S3, starts the stack, builds the
# policy index once, and gets the TLS certificate. Safe to re-run (settings are kept).
set -euo pipefail
if [ -z "${EIP:-}" ] && [ -f /etc/riskgraph/settings ]; then . /etc/riskgraph/settings; fi
: "${REGION:?}" "${EIP:?}" "${ECR_REGISTRY:?}" "${DATA_BUCKET:?}" "${SES_SENDER:?}" "${SES_RECIPIENT:?}"
cd /opt/riskgraph

if ! command -v docker >/dev/null; then # Docker Engine + Compose plugin from Docker's apt repo
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
        "https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        >/etc/apt/sources.list.d/docker.list
    apt-get update -q
    apt-get install -yq docker-ce docker-ce-cli containerd.io docker-compose-plugin jq
fi
command -v aws >/dev/null || snap install aws-cli --classic

mkdir -p /etc/riskgraph
printf '%s\n' "REGION=$REGION" "EIP=$EIP" "ECR_REGISTRY=$ECR_REGISTRY" "DATA_BUCKET=$DATA_BUCKET" \
    "SES_SENDER=$SES_SENDER" "SES_RECIPIENT=$SES_RECIPIENT" >/etc/riskgraph/settings

if [ ! -f data/processed/market_panel.parquet ]; then # DVC data from the S3 remote
    command -v uvx >/dev/null ||
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
    uvx --from 'dvc[s3]==3.67.1' dvc pull -r s3
fi
chown -R 1000:1000 data # the API container runs as uid 1000

# Basic auth for the site, from the app secret (the password goes through stdin, not argv).
app="$(aws secretsmanager get-secret-value --region "$REGION" --secret-id riskgraph/app \
    --query SecretString --output text)"
printf '%s:%s\n' "$(jq -r .BASIC_AUTH_USER <<<"$app")" \
    "$(jq -r .BASIC_AUTH_PASSWORD <<<"$app" | openssl passwd -apr1 -stdin)" >htpasswd
unset app

# A placeholder certificate lets nginx start; cert.sh swaps in the Let's Encrypt one.
mkdir -p certs certbot-www
[ -f certs/fullchain.pem ] || openssl req -x509 -nodes -newkey rsa:2048 -days 2 -subj "/CN=$EIP" \
    -keyout certs/privkey.pem -out certs/fullchain.pem 2>/dev/null

infra/aws/deploy.sh latest

if [ ! -f /etc/riskgraph/rag-indexed ]; then # policy index in the Weaviate volume
    docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm api \
        sh -c "python corpus/download.py && python -m riskgraph.cli rag index"
    touch /etc/riskgraph/rag-indexed
fi

infra/aws/cert.sh
cat >/etc/cron.d/riskgraph-cert <<'EOF'
17 3,15 * * * root /opt/riskgraph/infra/aws/cert.sh >>/var/log/riskgraph-cert.log 2>&1
@reboot root sleep 90 && /opt/riskgraph/infra/aws/cert.sh >>/var/log/riskgraph-cert.log 2>&1
EOF
echo "bootstrap done: https://$EIP"
