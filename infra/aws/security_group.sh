#!/usr/bin/env bash
# Security group for the instance (SPEC §13.2): HTTP and HTTPS from anywhere. No SSH port:
# shell access goes through SSM Session Manager, so port 22 stays closed.
source "$(dirname "$0")/common.sh"

vpc="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
plan "Security group $NAME-web in the default VPC ($vpc): inbound tcp/80 (HTTP -> HTTPS redirect and
Let's Encrypt validation) and tcp/443 (the site) from 0.0.0.0/0. No port 22 (use SSM Session
Manager). Outbound: all." \
    "\$0 (security groups are free)."

sg="$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME-web" "Name=vpc-id,Values=$vpc" \
    --query 'SecurityGroups[0].GroupId' --output text)"
if [[ $sg == None ]]; then
    sg="$(aws ec2 create-security-group --group-name "$NAME-web" --vpc-id "$vpc" \
        --description "RiskGraph web: 80 and 443 only" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$NAME-web},{Key=Project,Value=$NAME}]" \
        --query GroupId --output text)"
    echo "created: $sg"
fi
for port in 80 443; do
    aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port "$port" \
        --cidr 0.0.0.0/0 >/dev/null 2>&1 && echo "opened $port" || echo "already open: $port"
done
