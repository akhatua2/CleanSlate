#!/bin/bash
# Launch N spot instances in us-east-1 to build the IGM index over
# dolma3_mix-6T embarrassingly in parallel. Instance k builds shard k of N.
#
# Prereqs:
#   - aws-cli v2 configured with creds that can launch EC2 + write to S3
#   - ~/.zshrc.lfs sourced so $HF_TOKEN is set
#   - An IAM instance-profile ($ROLE_NAME) with AmazonS3FullAccess (or tighter)
#   - (Optional) EC2 keypair + security group for SSH debugging
#
# Usage:
#   source ~/.zshrc.lfs   # loads HF_TOKEN
#   NUM_SHARDS=28 ./launch_fleet.sh          # launches 28 instances
#   NUM_SHARDS=28 SHARDS="0,5,13" ./launch_fleet.sh   # just shards 0,5,13

set -euo pipefail
: "${HF_TOKEN:?source ~/.zshrc.lfs first to load HF_TOKEN}"

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-i4i.8xlarge}"
AMI_ID="${AMI_ID:-}"
ROLE_NAME="${ROLE_NAME:-cleanslate-ec2-role}"
KEY_NAME="${KEY_NAME:-}"
SG_ID="${SG_ID:-}"
GIT_REF="${GIT_REF:-main}"
NUM_SHARDS="${NUM_SHARDS:-28}"
SHARDS="${SHARDS:-}"          # comma-separated indices; empty = all 0..N-1
S3_BUCKET="${S3_BUCKET:-cleanslate-aws}"

if [ -z "$SHARDS" ]; then
  SHARDS=$(seq -s, 0 $((NUM_SHARDS - 1)))
fi

# Resolve latest Ubuntu 24.04 AMI via SSM if not given
if [ -z "$AMI_ID" ]; then
  AMI_ID=$(aws ssm get-parameter --region "$REGION" \
    --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query 'Parameter.Value' --output text)
fi
echo "AMI=$AMI_ID  TYPE=$INSTANCE_TYPE  REGION=$REGION"
echo "NUM_SHARDS=$NUM_SHARDS  SHARDS=$SHARDS"

USER_DATA_FILE="$(dirname "$0")/user_data.sh"
test -f "$USER_DATA_FILE"

launch_one () {
  local idx="$1"
  local rendered
  rendered=$(mktemp)
  trap 'rm -f "$rendered"' RETURN
  {
    echo "#!/bin/bash"
    echo "export HF_TOKEN='$HF_TOKEN'"
    echo "export S3_BUCKET='$S3_BUCKET'"
    echo "export GIT_REF='$GIT_REF'"
    echo "export NUM_SHARDS='$NUM_SHARDS'"
    echo "export SHARD_INDEX='$idx'"
    tail -n +2 "$USER_DATA_FILE"
  } > "$rendered"

  local args=(
    --region "$REGION"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --count 1
    --iam-instance-profile "Name=$ROLE_NAME"
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'
    --instance-initiated-shutdown-behavior terminate
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cleanslate-igm-shard-$idx},{Key=Project,Value=cleanslate},{Key=ShardIndex,Value=$idx}]"
    --user-data "file://$rendered"
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOnTermination=true}'
  )
  [ -n "$KEY_NAME" ] && args+=( --key-name "$KEY_NAME" )
  [ -n "$SG_ID" ]    && args+=( --security-group-ids "$SG_ID" )

  echo "launching shard $idx"
  aws ec2 run-instances "${args[@]}" \
    --query 'Instances[0].InstanceId' --output text
}

IFS=',' read -ra INDICES <<< "$SHARDS"
for idx in "${INDICES[@]}"; do
  launch_one "$idx"
done

echo
echo "Monitoring:"
echo "  aws s3 ls s3://$S3_BUCKET/igm-dolma6T/ --recursive --human-readable"
echo "  aws ec2 describe-instances --region $REGION --filters 'Name=tag:Project,Values=cleanslate' 'Name=instance-state-name,Values=running' --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==\`ShardIndex\`].Value|[0],State.Name]' --output table"
