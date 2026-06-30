#!/bin/bash
# Deploy the emoji generator API to Cloud Run.
# Run from repo root: bash api/deploy.sh

set -euo pipefail

PROJECT="emoji-generator-69"
REGION="us-central1"
SERVICE="emoji-generator-api"
IMAGE="gcr.io/${PROJECT}/${SERVICE}"

echo "Building image..."
gcloud builds submit \
  --project "$PROJECT" \
  --tag "$IMAGE" \
  --dockerfile api/Dockerfile \
  .

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 120 \
  --set-env-vars "MODEL_CKPT=emoji_fm_best.pt,GUIDANCE_SCALE=7.5" \
  --set-env-vars "CORS_ORIGINS=https://emoji-generator-69.web.app"

echo ""
echo "Deploy complete. Set NEXT_PUBLIC_API_URL in emoji-generator/web/.env.local to:"
gcloud run services describe "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --format "value(status.url)"
