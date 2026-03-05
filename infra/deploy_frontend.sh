#!/usr/bin/env bash
# Deploys Flask frontend to Azure Container Apps (consumption profile, scale-to-zero).

set -euo pipefail
export PYTHONIOENCODING="utf-8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: .env not found at ${ENV_FILE}"
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

RG="${AZURE_RESOURCE_GROUP}"
LOCATION="${AZURE_LOCATION:-$(az group show --name "${RG}" --query location -o tsv)}"

APP_NAME="${AZURE_FRONTEND_APP_NAME:-financial-agent-web}"
ACA_ENV_NAME="${AZURE_CONTAINERAPPS_ENV:-financial-agent-env}"
LOG_NAME="${AZURE_LOG_WORKSPACE:-financial-agent-logs}"
ACR_NAME="${AZURE_ACR_NAME:-}"
if [[ -z "${ACR_NAME}" ]]; then
  # ACR names must be global, 5-50 chars, lowercase alphanumeric.
  SUFFIX="$(echo -n "${RG}" | tr -cd 'a-z0-9' | tail -c 8)"
  ACR_NAME="finagentacr${SUFFIX}"
fi

IMAGE_NAME="financial-analyst-agent-web"
IMAGE_TAG="$(date +%Y%m%d%H%M%S)"

echo "Resource group: ${RG}"
echo "Location      : ${LOCATION}"
echo "Container App : ${APP_NAME}"
echo "ACA Env       : ${ACA_ENV_NAME}"
echo "ACR           : ${ACR_NAME}"

az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait --only-show-errors
az provider register --namespace Microsoft.OperationalInsights --wait --only-show-errors
az provider register --namespace Microsoft.ContainerRegistry --wait --only-show-errors

if ! az monitor log-analytics workspace show --resource-group "${RG}" --workspace-name "${LOG_NAME}" --output none 2>/dev/null; then
  az monitor log-analytics workspace create \
    --resource-group "${RG}" \
    --workspace-name "${LOG_NAME}" \
    --location "${LOCATION}" \
    --output none
fi

LOG_ID="$(az monitor log-analytics workspace show \
  --resource-group "${RG}" \
  --workspace-name "${LOG_NAME}" \
  --query customerId -o tsv)"
LOG_KEY="$(az monitor log-analytics workspace get-shared-keys \
  --resource-group "${RG}" \
  --workspace-name "${LOG_NAME}" \
  --query primarySharedKey -o tsv)"

if ! az containerapp env show --name "${ACA_ENV_NAME}" --resource-group "${RG}" --output none 2>/dev/null; then
  az containerapp env create \
    --name "${ACA_ENV_NAME}" \
    --resource-group "${RG}" \
    --location "${LOCATION}" \
    --logs-workspace-id "${LOG_ID}" \
    --logs-workspace-key "${LOG_KEY}" \
    --output none
fi

if ! az acr show --name "${ACR_NAME}" --resource-group "${RG}" --output none 2>/dev/null; then
  az acr create \
    --name "${ACR_NAME}" \
    --resource-group "${RG}" \
    --location "${LOCATION}" \
    --sku Basic \
    --admin-enabled true \
    --output none
fi

REGISTRY_SERVER="$(az acr show --name "${ACR_NAME}" --resource-group "${RG}" --query loginServer -o tsv)"
REGISTRY_USER="$(az acr credential show --name "${ACR_NAME}" --resource-group "${RG}" --query username -o tsv)"
REGISTRY_PASS="$(az acr credential show --name "${ACR_NAME}" --resource-group "${RG}" --query passwords[0].value -o tsv)"
FULL_IMAGE="${REGISTRY_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building and pushing image: ${FULL_IMAGE}"
az acr build \
  --registry "${ACR_NAME}" \
  --image "${IMAGE_NAME}:${IMAGE_TAG}" \
  --file "${PROJECT_ROOT}/Dockerfile" \
  --no-logs \
  "${PROJECT_ROOT}" \
  --output none

if ! az containerapp show --name "${APP_NAME}" --resource-group "${RG}" --output none 2>/dev/null; then
  az containerapp create \
    --name "${APP_NAME}" \
    --resource-group "${RG}" \
    --environment "${ACA_ENV_NAME}" \
    --image "${FULL_IMAGE}" \
    --ingress external \
    --target-port 8000 \
    --registry-server "${REGISTRY_SERVER}" \
    --registry-username "${REGISTRY_USER}" \
    --registry-password "${REGISTRY_PASS}" \
    --cpu 0.25 \
    --memory 0.5Gi \
    --min-replicas 0 \
    --max-replicas 1 \
    --secrets \
      aoai-key="${AZURE_OPENAI_KEY}" \
      search-key="${AZURE_SEARCH_KEY}" \
      storage-conn="${AZURE_STORAGE_CONNECTION_STRING}" \
    --env-vars \
      AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}" \
      AZURE_OPENAI_KEY=secretref:aoai-key \
      AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}" \
      AZURE_OPENAI_EMBEDDING_DEPLOYMENT="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}" \
      AZURE_OPENAI_LLM_DEPLOYMENT="${AZURE_OPENAI_LLM_DEPLOYMENT}" \
      AZURE_SEARCH_ENDPOINT="${AZURE_SEARCH_ENDPOINT}" \
      AZURE_SEARCH_KEY=secretref:search-key \
      AZURE_SEARCH_INDEX_NAME="${AZURE_SEARCH_INDEX_NAME}" \
      AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn \
      AZURE_STORAGE_CONTAINER="${AZURE_STORAGE_CONTAINER}" \
    --output none
else
  az containerapp secret set \
    --name "${APP_NAME}" \
    --resource-group "${RG}" \
    --secrets \
      aoai-key="${AZURE_OPENAI_KEY}" \
      search-key="${AZURE_SEARCH_KEY}" \
      storage-conn="${AZURE_STORAGE_CONNECTION_STRING}" \
    --output none

  az containerapp registry set \
    --name "${APP_NAME}" \
    --resource-group "${RG}" \
    --server "${REGISTRY_SERVER}" \
    --username "${REGISTRY_USER}" \
    --password "${REGISTRY_PASS}" \
    --output none

  az containerapp update \
    --name "${APP_NAME}" \
    --resource-group "${RG}" \
    --image "${FULL_IMAGE}" \
    --set-env-vars \
      AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}" \
      AZURE_OPENAI_KEY=secretref:aoai-key \
      AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}" \
      AZURE_OPENAI_EMBEDDING_DEPLOYMENT="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}" \
      AZURE_OPENAI_LLM_DEPLOYMENT="${AZURE_OPENAI_LLM_DEPLOYMENT}" \
      AZURE_SEARCH_ENDPOINT="${AZURE_SEARCH_ENDPOINT}" \
      AZURE_SEARCH_KEY=secretref:search-key \
      AZURE_SEARCH_INDEX_NAME="${AZURE_SEARCH_INDEX_NAME}" \
      AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn \
      AZURE_STORAGE_CONTAINER="${AZURE_STORAGE_CONTAINER}" \
    --min-replicas 0 \
    --max-replicas 1 \
    --cpu 0.25 \
    --memory 0.5Gi \
    --output none
fi

APP_URL="https://$(az containerapp show --name "${APP_NAME}" --resource-group "${RG}" --query properties.configuration.ingress.fqdn -o tsv)"
echo "Deployment complete."
echo "Frontend URL: ${APP_URL}"
