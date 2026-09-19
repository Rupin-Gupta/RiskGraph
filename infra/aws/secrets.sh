#!/usr/bin/env bash
# Secrets Manager entries (SPEC §13.2). --apply creates the names with empty values; you enter
# the values in the console, or run with --apply --load-from-env to copy them from .env. Values
# are never printed. The Postgres password, approval secret, and API token are generated once
# for production and never rotated by this script (a new Postgres password would lock out the
# existing database volume).
source "$(dirname "$0")/common.sh"
KEYS=(POSTGRES_PASSWORD APPROVAL_SECRET GEMINI_API_KEY FRED_API_KEY LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY LANGFUSE_HOST LANGFUSE_PROJECT_ID BASIC_AUTH_USER BASIC_AUTH_PASSWORD)

plan "Secret $APP_SECRET: JSON with keys ${KEYS[*]}.
Secret $TOKEN_SECRET: the bearer token for POST /runs/daily (read by the trigger Lambda only).
With --load-from-env: keys from .env where set; POSTGRES_PASSWORD, APPROVAL_SECRET, and the API
token are generated once if empty." \
    "\$0.40 per secret per month: \$0.80/month (API calls \$0.05 per 10,000)."

for s in "$APP_SECRET" "$TOKEN_SECRET"; do
    if aws secretsmanager describe-secret --secret-id "$s" >/dev/null 2>&1; then
        echo "exists: $s"
    else
        init='""'
        [[ $s == "$APP_SECRET" ]] && init="$(printf '"%s":"",' "${KEYS[@]}")" && init="{${init%,}}"
        aws secretsmanager create-secret --name "$s" --secret-string "$init" \
            --tags "Key=Project,Value=$NAME" >/dev/null
        echo "created: $s (empty values)"
    fi
done

[[ " $* " == *" --load-from-env "* ]] || exit 0

current() { aws secretsmanager get-secret-value --secret-id "$1" --query SecretString --output text; }
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
chmod 600 "$tmp"
# Build the JSON in Python: values stay in memory and the temp file, never on the terminal.
current "$APP_SECRET" | ENV_FILE="$ROOT/.env" KEYS="${KEYS[*]}" python3 -c '
import json, os, secrets, sys
cur = json.loads(sys.stdin.read() or "{}")
env = {}
for line in open(os.environ["ENV_FILE"]):
    k, sep, v = line.rstrip("\n").partition("=")
    if sep and not k.startswith("#"):
        env[k.strip()] = v.strip()
out = {}
for k in os.environ["KEYS"].split():
    out[k] = env.get(k) or cur.get(k, "")
for k in ("POSTGRES_PASSWORD", "APPROVAL_SECRET"):
    out[k] = cur.get(k) or secrets.token_hex(24)  # production values, generated once
print(json.dumps(out))
missing = [k for k, v in out.items() if not v]
print("still empty:", ", ".join(missing) or "none", file=sys.stderr)
' >"$tmp"
aws secretsmanager put-secret-value --secret-id "$APP_SECRET" --secret-string "file://$tmp" >/dev/null
echo "updated: $APP_SECRET"

if [[ -z "$(current "$TOKEN_SECRET" | tr -d '"')" ]]; then
    python3 -c 'import secrets; print(secrets.token_hex(24), end="")' >"$tmp"
    aws secretsmanager put-secret-value --secret-id "$TOKEN_SECRET" --secret-string "file://$tmp" >/dev/null
    echo "generated: $TOKEN_SECRET"
fi
