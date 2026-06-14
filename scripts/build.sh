set -e  # Exit on any error

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-east-1"
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
VERSION=$(git rev-parse --short HEAD)  # Use git commit SHA as image tag

echo "=== Deploying RAG System ==="
echo "Account: $ACCOUNT_ID | Region: $REGION | Version: $VERSION"

# Login to ECR
echo "--- Logging into ECR ---"
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

# Build and push API
echo "--- Building API image ---"
docker build -t rag-api:$VERSION ./services/api
docker tag rag-api:$VERSION $ECR_REGISTRY/rag-api:$VERSION
docker tag rag-api:$VERSION $ECR_REGISTRY/rag-api:latest
docker push $ECR_REGISTRY/rag-api:$VERSION
docker push $ECR_REGISTRY/rag-api:latest

# Update deployment with new image
# This triggers a rolling update — zero downtime!
echo "--- Updating Kubernetes deployment ---"
kubectl set image deployment/rag-api-deployment \
    rag-api=$ECR_REGISTRY/rag-api:$VERSION \
    -n rag-system

# Wait for rollout to complete
kubectl rollout status deployment/rag-api-deployment -n rag-system

echo "=== Deployment complete! ==="
echo "API available at: http://$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')/docs"