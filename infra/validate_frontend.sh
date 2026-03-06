#!/usr/bin/env bash
# Pre-deployment checks for the Flask frontend on Azure Container Apps.

set -euo pipefail

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

required_vars=(
  AZURE_RESOURCE_GROUP
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_KEY
  AZURE_OPENAI_API_VERSION
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT
  AZURE_OPENAI_LLM_DEPLOYMENT
  AZURE_SEARCH_ENDPOINT
  AZURE_SEARCH_KEY
  AZURE_SEARCH_INDEX_NAME
  AZURE_STORAGE_CONNECTION_STRING
  AZURE_STORAGE_CONTAINER
)

echo "Running validation checks..."
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: Missing ${name} in .env"
    exit 1
  fi
done

az account show --output none
az group show --name "${AZURE_RESOURCE_GROUP}" --output none

az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait --only-show-errors
az provider register --namespace Microsoft.OperationalInsights --wait --only-show-errors
az provider register --namespace Microsoft.ContainerRegistry --wait --only-show-errors

echo "Validation successful."

