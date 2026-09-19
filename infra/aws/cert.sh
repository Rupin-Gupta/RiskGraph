#!/usr/bin/env bash
# Issue or renew the Let's Encrypt certificate for the Elastic IP, then reload nginx (root, on
# the host). IP certificates need the shortlived profile (~6 days); certbot renews at half the
# lifetime, so the twice-daily cron issues about two certificates a week (ADR-014).
set -euo pipefail
. /etc/riskgraph/settings
cd /opt/riskgraph

docker run --rm -v /etc/letsencrypt:/etc/letsencrypt -v "$PWD/certbot-www:/var/www/certbot" \
    certbot/certbot:v5.8.0 certonly -n -q --agree-tos --register-unsafely-without-email \
    --webroot -w /var/www/certbot --preferred-profile shortlived --ip-address "$EIP" \
    --cert-name riskgraph
install -m 600 /etc/letsencrypt/live/riskgraph/privkey.pem certs/privkey.pem
install -m 644 /etc/letsencrypt/live/riskgraph/fullchain.pem certs/fullchain.pem
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T proxy nginx -s reload
