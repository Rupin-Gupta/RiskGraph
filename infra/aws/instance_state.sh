#!/usr/bin/env bash
# make aws-stop / make aws-start: stop or start the instance to control spend. The Elastic IP and
# the EBS volume keep billing while it is stopped (about $6.40/month together).
source "$(dirname "$0")/common.sh"
action="${1:-}"
[[ $action == start || $action == stop ]] || { echo "usage: $0 start|stop [--apply]" >&2; exit 1; }

id="$(instance_id)"
[[ -n $id ]] || { echo "no $NAME instance" >&2; exit 1; }
now="$(aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"

plan "$action instance $id (currently $now)." \
    "$([[ $action == stop ]] &&
        echo 'saves $0.1008/hour of compute; the Elastic IP ($3.65/month) and the 30 GB volume ($2.74/month) keep billing' ||
        echo '$0.1008/hour while running, plus the Elastic IP and volume')"

if [[ $action == start ]]; then
    aws ec2 start-instances --instance-ids "$id" >/dev/null
    aws ec2 wait instance-running --instance-ids "$id"
    echo "started. The stack restarts on its own (restart: unless-stopped); give it a minute."
    echo "site: https://$(eip)"
else
    aws ec2 stop-instances --instance-ids "$id" >/dev/null
    aws ec2 wait instance-stopped --instance-ids "$id"
    echo "stopped. The weekday trigger will skip until you start it again."
fi
