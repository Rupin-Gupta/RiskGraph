#!/usr/bin/env bash
# S3 buckets: the DVC remote and the MLflow/model artifacts store (SPEC §13.2).
source "$(dirname "$0")/common.sh"

plan "S3 buckets $DATA_BUCKET (DVC remote) and $ARTIFACTS_BUCKET (MLflow artifacts, GGUF model).
All public access blocked; default SSE-S3 encryption; tagged Project=$NAME." \
    "S3 Standard \$0.025/GB-month: about \$0.01/month for today's ~3 MB of data."

for b in "$DATA_BUCKET" "$ARTIFACTS_BUCKET"; do
    if aws s3api head-bucket --bucket "$b" 2>/dev/null; then
        echo "exists: $b"
    else
        aws s3api create-bucket --bucket "$b" \
            --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
        echo "created: $b"
    fi
    aws s3api put-public-access-block --bucket "$b" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    aws s3api put-bucket-tagging --bucket "$b" --tagging "TagSet=[{Key=Project,Value=$NAME}]"
done
