#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set default values
PROJECT_ID=""
IMAGE_NAME="indonesian-cooking-assistant"
REGION="asia-southeast1"
MEMORY="32Gi"
HF_TOKEN="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"

echo -e "${YELLOW}=====================================================
DEPLOYMENT SCRIPT FOR GOOGLE CLOUD RUN (SOUTHEAST ASIA)
=====================================================${NC}"

# Check if required files exist
echo -e "\n${GREEN}Checking required files...${NC}"
for file in "food.py" "requirements.txt" "Dockerfile" ".dockerignore"; do
    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}Warning: $file not found. Make sure it exists before continuing.${NC}"
    else
        echo "✓ $file found"
    fi
done

# Get Hugging Face token
# echo -e "\n${GREEN}Hugging Face Authentication${NC}"
# read -p "Do you need to use a Hugging Face token? [y/N]: " NEED_HF_TOKEN
# if [[ $NEED_HF_TOKEN =~ ^[Yy]$ ]]; then
#     read -p "Enter your Hugging Face token: " HF_TOKEN
#     echo "Hugging Face token will be used for authentication."
# else
#     echo "No Hugging Face token will be used."
# fi

# Get Google Cloud Project ID
if [ -z "$PROJECT_ID" ]; then
    echo -e "\n${GREEN}Getting Google Cloud Project ID...${NC}"
    # Try to get the current project
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    
    if [ -z "$CURRENT_PROJECT" ] || [ "$CURRENT_PROJECT" = "(unset)" ]; then
        echo -e "${YELLOW}No project currently set.${NC}"
        read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    else
        echo "Current project: $CURRENT_PROJECT"
        read -p "Use this project? [Y/n]: " USE_CURRENT
        if [[ $USE_CURRENT =~ ^[Nn]$ ]]; then
            read -p "Enter your Google Cloud Project ID: " PROJECT_ID
        else
            PROJECT_ID=$CURRENT_PROJECT
        fi
    fi
    
    # Set the project
    echo "Setting Google Cloud project to: $PROJECT_ID"
    gcloud config set project $PROJECT_ID
fi

# Authenticate with Google Cloud if needed
echo -e "\n${GREEN}Checking authentication...${NC}"
gcloud auth print-access-token &>/dev/null
if [ $? -ne 0 ]; then
    echo "You need to authenticate with Google Cloud"
    gcloud auth login
else
    echo "✓ Already authenticated with Google Cloud"
fi

# Configure Docker for GCR
echo -e "\n${GREEN}Configuring Docker for Google Container Registry...${NC}"
gcloud auth configure-docker

# Build Docker image
echo -e "\n${GREEN}Building Docker image...${NC}"
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Push to Google Container Registry
echo -e "\n${GREEN}Pushing image to Google Container Registry...${NC}"
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
echo -e "\n${GREEN}Deploying to Cloud Run in $REGION...${NC}"
DEPLOY_CMD="gcloud run deploy $IMAGE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory $MEMORY \
  --cpu 16 \
  --timeout=1800 \
  --min-instances=1"

# Add Hugging Face token as environment variable if provided
if [ -n "$HF_TOKEN" ]; then
    DEPLOY_CMD="$DEPLOY_CMD --set-env-vars=HF_TOKEN=$HF_TOKEN"
fi

# Execute the deployment command
eval $DEPLOY_CMD

# Get the URL of the deployed service
echo -e "\n${GREEN}Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe $IMAGE_NAME --region $REGION --format='value(status.url)')

echo -e "\n${GREEN}Deployment complete!${NC}"
echo -e "Your application is available at: ${YELLOW}$SERVICE_URL${NC}"