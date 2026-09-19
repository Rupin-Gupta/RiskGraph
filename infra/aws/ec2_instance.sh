#!/usr/bin/env bash
# The EC2 host (SPEC §13.2, ADR-014): one m7i-flex.large on Ubuntu 24.04 with a 30 GB gp3 root
# volume, IMDSv2 only, and the Elastic IP. User data clones the public repo and runs
# infra/aws/bootstrap.sh, which installs Docker, pulls the data and images, and starts the stack.
source "$(dirname "$0")/common.sh"
: "${SES_SENDER:?set SES_SENDER in .env}" "${SES_RECIPIENT:?set SES_RECIPIENT in .env}"
TYPE=m7i-flex.large

ip="$(eip)"
[[ $ip != None ]] || { echo "run elastic_ip.sh --apply first" >&2; exit 1; }
existing="$(instance_id)"
plan "Instance $NAME: $TYPE (2 vCPU, 8 GB; Free-plan eligible), Ubuntu 24.04 LTS (latest Canonical
AMI), 30 GB gp3 encrypted root volume, IMDSv2 required (hop limit 2 so containers get role
credentials), instance profile $NAME-ec2, security group $NAME-web, Elastic IP $ip.
User data: clone https://github.com/$GITHUB_REPO and run infra/aws/bootstrap.sh
(Docker, DVC pull from $DATA_BUCKET, secrets -> .env, images from ECR, RAG index, TLS).
${existing:+Already exists: $existing (nothing new is launched; the EIP is re-associated).}" \
    "\$0.1008/hour while running (\$2.42/day, \$73.55/month if never stopped) + gp3 30 GB at
\$0.0912/GB-month = \$2.74/month even when stopped. Stop it when idle: make aws-stop."

if [[ -z $existing ]]; then
    ami="$(aws ssm get-parameter --query Parameter.Value --output text \
        --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id)"
    sg="$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME-web" \
        --query 'SecurityGroups[0].GroupId' --output text)"
    userdata="$(mktemp)"
    trap 'rm -f "$userdata"' EXIT
    cat >"$userdata" <<EOF
#!/bin/bash
set -euo pipefail
exec >>/var/log/riskgraph-bootstrap.log 2>&1
git clone https://github.com/$GITHUB_REPO.git /opt/riskgraph
REGION=$REGION EIP=$ip ECR_REGISTRY=$ECR_REGISTRY DATA_BUCKET=$DATA_BUCKET \
  SES_SENDER=$SES_SENDER SES_RECIPIENT=$SES_RECIPIENT /opt/riskgraph/infra/aws/bootstrap.sh
EOF
    tags="Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$NAME}]"
    existing="$(aws ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --count 1 \
        --iam-instance-profile "Name=$NAME-ec2" --security-group-ids "$sg" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}' \
        --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
        --user-data "file://$userdata" \
        --tag-specifications "ResourceType=instance,$tags" "ResourceType=volume,$tags" \
        --query 'Instances[0].InstanceId' --output text)"
    echo "launched: $existing ($ami)"
fi
aws ec2 wait instance-running --instance-ids "$existing"
alloc="$(aws ec2 describe-addresses --public-ips "$ip" --query 'Addresses[0].AllocationId' --output text)"
aws ec2 associate-address --allocation-id "$alloc" --instance-id "$existing" --allow-reassociation >/dev/null
echo "running with $ip. Bootstrap takes ~10 minutes; follow it with:"
echo "  aws ssm start-session --target $existing   # then: sudo tail -f /var/log/riskgraph-bootstrap.log"
echo "site: https://$ip"
