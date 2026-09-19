#!/usr/bin/env bash
# Elastic IP: a stable address for the site, its Let's Encrypt certificate, and the trigger
# Lambda. Associates it with the instance once the instance exists (ADR-014).
source "$(dirname "$0")/common.sh"

plan "Elastic IP tagged Name=$NAME (allocated once), associated with the $NAME instance if it exists." \
    "Public IPv4 \$0.005/hour whether the instance runs or not: \$3.65/month."

ip="$(eip)"
if [[ $ip == None ]]; then
    ip="$(aws ec2 allocate-address --domain vpc \
        --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$NAME}]" \
        --query PublicIp --output text)"
    echo "allocated: $ip"
else
    echo "exists: $ip"
fi
id="$(instance_id)"
if [[ -n $id ]]; then
    alloc="$(aws ec2 describe-addresses --public-ips "$ip" --query 'Addresses[0].AllocationId' --output text)"
    aws ec2 associate-address --allocation-id "$alloc" --instance-id "$id" --allow-reassociation >/dev/null
    echo "associated with $id"
fi
echo "site URL: https://$ip"
