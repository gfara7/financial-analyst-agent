#!/usr/bin/env bash
# infra/provision.sh
# Provisions all Azure resources for the Financial Analyst Agent MVP
#
# Usage:
#   bash infra/provision.sh [resource-group] [location]
#
# Example:
#   bash infra/provision.sh rg-financial-agent swedencentral
#
# Requirements:
#   - Azure CLI installed and logged in (az login)
#   - Sufficient subscription quota for Azure OpenAI (request via portal if needed)
#
# Re-run safe: existing resources are reused, not recreated.
# The resource suffix is persisted in infra/.suffix so names stay consistent.
#
# Estimated cost: ~$2.50/day (Basic Search + AOAI usage)
# Tear down with: bash infra/teardown.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Configuration ─────────────────────────────────────────────────────────────
RG="${1:-rg-financial-agent}"
LOCATION="${2:-swedencentral}"
STORAGE_CONTAINER="source-pdfs"

# Persist suffix so re-runs reuse the same resource names
SUFFIX_FILE="${SCRIPT_DIR}/.suffix"
if [[ -f "$SUFFIX_FILE" ]]; then
    UNIQUE_SUFFIX=$(tr -d '[:space:]' < "$SUFFIX_FILE")
    echo "      Reusing existing suffix: $UNIQUE_SUFFIX"
else
    UNIQUE_SUFFIX=$(openssl rand -hex 3)
    echo "$UNIQUE_SUFFIX" > "$SUFFIX_FILE"
    echo "      Generated new suffix: $UNIQUE_SUFFIX"
fi

AOAI_NAME="aoai-fin-agent-${UNIQUE_SUFFIX}"
SEARCH_NAME="srch-fin-agent-${UNIQUE_SUFFIX}"
STORAGE_NAME="stfinagent${UNIQUE_SUFFIX}"

echo "============================================================"
echo " Financial Analyst Agent — Azure Provisioning"
echo "============================================================"
echo " Resource Group : $RG"
echo " Location       : $LOCATION"
echo " Suffix         : $UNIQUE_SUFFIX"
echo "============================================================"
echo ""

# ── 0. Subscription ───────────────────────────────────────────────────────────
echo "[0/6] Setting up Azure subscription..."
SUBSCRIPTION_ID=""

if az account alias --help > /dev/null 2>&1; then
    BILLING_ACCOUNT_ID=$(az billing account list \
        --query "[0].name" -o tsv 2>/dev/null || echo "")

    if [ -n "$BILLING_ACCOUNT_ID" ]; then
        AGREEMENT_TYPE=$(az billing account show \
            --name "$BILLING_ACCOUNT_ID" \
            --query "agreementType" -o tsv 2>/dev/null || echo "")

        if [ "$AGREEMENT_TYPE" = "EnterpriseAgreement" ]; then
            ENROLLMENT_ACCOUNT_ID=$(az billing enrollment-account list \
                --billing-account-name "$BILLING_ACCOUNT_ID" \
                --query "[0].name" -o tsv 2>/dev/null || echo "")
            if [ -n "$ENROLLMENT_ACCOUNT_ID" ]; then
                BILLING_SCOPE="/providers/Microsoft.Billing/billingAccounts/${BILLING_ACCOUNT_ID}/enrollmentAccounts/${ENROLLMENT_ACCOUNT_ID}"
                SUBSCRIPTION_ID=$(az account alias create \
                    --name "financial-agent-dev-${UNIQUE_SUFFIX}" \
                    --billing-scope "$BILLING_SCOPE" \
                    --workload "DevTest" \
                    --query "properties.subscriptionId" -o tsv 2>/dev/null || echo "")
            fi
        elif [ "$AGREEMENT_TYPE" = "MicrosoftCustomerAgreement" ]; then
            BILLING_PROFILE_ID=$(az billing profile list \
                --account-name "$BILLING_ACCOUNT_ID" \
                --query "[0].name" -o tsv 2>/dev/null || echo "")
            INVOICE_SECTION_ID=$(az billing invoice section list \
                --account-name "$BILLING_ACCOUNT_ID" \
                --profile-name "$BILLING_PROFILE_ID" \
                --query "[0].name" -o tsv 2>/dev/null || echo "")
            if [ -n "$BILLING_PROFILE_ID" ] && [ -n "$INVOICE_SECTION_ID" ]; then
                BILLING_SCOPE="/providers/Microsoft.Billing/billingAccounts/${BILLING_ACCOUNT_ID}/billingProfiles/${BILLING_PROFILE_ID}/invoiceSections/${INVOICE_SECTION_ID}"
                SUBSCRIPTION_ID=$(az account alias create \
                    --name "financial-agent-dev-${UNIQUE_SUFFIX}" \
                    --billing-scope "$BILLING_SCOPE" \
                    --workload "Production" \
                    --query "properties.subscriptionId" -o tsv 2>/dev/null || echo "")
            fi
        fi
    fi
fi

if [ -n "$SUBSCRIPTION_ID" ]; then
    echo "      New subscription: $SUBSCRIPTION_ID"
    sleep 30
    az account set --subscription "$SUBSCRIPTION_ID"
else
    echo "      Using current active subscription."
    SUBSCRIPTION_ID=$(az account show --query "id" -o tsv | tr -d '[:space:]')
    SUB_NAME=$(az account show --query "name" -o tsv | tr -d '\r')
    echo "      $SUB_NAME ($SUBSCRIPTION_ID)"
fi

echo ""

# ── 1. Resource providers ─────────────────────────────────────────────────────
echo "[1/6] Registering resource providers..."
for PROVIDER in "Microsoft.CognitiveServices" "Microsoft.Search" "Microsoft.Storage"; do
    az provider register --namespace "$PROVIDER" --wait --output none 2>/dev/null || true
done
echo "      Done."

# ── 2. Resource Group ─────────────────────────────────────────────────────────
echo ""
echo "[2/6] Resource group: $RG"
az group create --name "$RG" --location "$LOCATION" --output table

# ── 3. Azure OpenAI ───────────────────────────────────────────────────────────
echo ""
echo "[3/6] Azure OpenAI account: $AOAI_NAME"

if az cognitiveservices account show --name "$AOAI_NAME" --resource-group "$RG" --output none 2>/dev/null; then
    echo "      Already exists — skipping creation."
else
    az cognitiveservices account create \
        --name "$AOAI_NAME" \
        --resource-group "$RG" \
        --location "$LOCATION" \
        --kind OpenAI \
        --sku S0 \
        --yes \
        --output table
fi

echo "      Deploying text-embedding-ada-002..."
if az cognitiveservices account deployment show \
    --name "$AOAI_NAME" --resource-group "$RG" \
    --deployment-name "text-embedding-ada-002" --output none 2>/dev/null; then
    echo "      Already deployed — skipping."
else
    az cognitiveservices account deployment create \
        --name "$AOAI_NAME" \
        --resource-group "$RG" \
        --deployment-name "text-embedding-ada-002" \
        --model-name "text-embedding-ada-002" \
        --model-version "2" \
        --model-format OpenAI \
        --sku-capacity 100 \
        --sku-name "Standard" \
        --output table
fi

echo "      Deploying gpt-4o..."
if az cognitiveservices account deployment show \
    --name "$AOAI_NAME" --resource-group "$RG" \
    --deployment-name "gpt-4o" --output none 2>/dev/null; then
    echo "      Already deployed — skipping."
else
    az cognitiveservices account deployment create \
        --name "$AOAI_NAME" \
        --resource-group "$RG" \
        --deployment-name "gpt-4o" \
        --model-name "gpt-4o" \
        --model-version "2024-08-06" \
        --model-format OpenAI \
        --sku-capacity 30 \
        --sku-name "Standard" \
        --output table
fi

AOAI_ENDPOINT=$(az cognitiveservices account show \
    --name "$AOAI_NAME" --resource-group "$RG" \
    --query "properties.endpoint" -o tsv | tr -d '[:space:]')

AOAI_KEY=$(az cognitiveservices account keys list \
    --name "$AOAI_NAME" --resource-group "$RG" \
    --query "key1" -o tsv | tr -d '[:space:]')

# ── 4. Azure AI Search ────────────────────────────────────────────────────────
echo ""
echo "[4/6] Azure AI Search: $SEARCH_NAME (Basic tier)"

if az search service show --name "$SEARCH_NAME" --resource-group "$RG" --output none 2>/dev/null; then
    echo "      Already exists — skipping creation."
else
    az search service create \
        --name "$SEARCH_NAME" \
        --resource-group "$RG" \
        --location "$LOCATION" \
        --sku basic \
        --replica-count 1 \
        --partition-count 1 \
        --output table
fi

SEARCH_ENDPOINT="https://${SEARCH_NAME}.search.windows.net"
SEARCH_KEY=$(az search admin-key show \
    --service-name "$SEARCH_NAME" --resource-group "$RG" \
    --query "primaryKey" -o tsv | tr -d '[:space:]')

# ── 5. Azure Blob Storage ─────────────────────────────────────────────────────
echo ""
echo "[5/6] Storage account: $STORAGE_NAME"

if az storage account show --name "$STORAGE_NAME" --resource-group "$RG" --output none 2>/dev/null; then
    echo "      Already exists — skipping creation."
else
    az storage account create \
        --name "$STORAGE_NAME" \
        --resource-group "$RG" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2 \
        --output table
fi

STORAGE_CONN=$(az storage account show-connection-string \
    --name "$STORAGE_NAME" --resource-group "$RG" \
    --query "connectionString" -o tsv | tr -d '[:space:]')

az storage container create \
    --name "$STORAGE_CONTAINER" \
    --connection-string "$STORAGE_CONN" \
    --output table

# ── 6. Write .env ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Writing .env..."

cat > "${PROJECT_ROOT}/.env" <<EOF
# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT=${AOAI_ENDPOINT}
AZURE_OPENAI_KEY=${AOAI_KEY}
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o

# ── Azure AI Search ───────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT=${SEARCH_ENDPOINT}
AZURE_SEARCH_KEY=${SEARCH_KEY}
AZURE_SEARCH_INDEX_NAME=financial-docs

# ── Azure Blob Storage ────────────────────────────────────────────────────────
AZURE_STORAGE_CONNECTION_STRING=${STORAGE_CONN}
AZURE_STORAGE_CONTAINER=${STORAGE_CONTAINER}

# ── Resource tracking (for teardown.sh) ──────────────────────────────────────
AZURE_SUBSCRIPTION_ID=${SUBSCRIPTION_ID}
AZURE_SUBSCRIPTION_NAME=${SUB_NAME:-current}
AZURE_RESOURCE_GROUP=${RG}
AZURE_AOAI_NAME=${AOAI_NAME}
AZURE_SEARCH_NAME=${SEARCH_NAME}
AZURE_STORAGE_NAME=${STORAGE_NAME}
EOF

echo ""
echo "============================================================"
echo " Provisioning complete!"
echo "============================================================"
echo " AOAI endpoint   : $AOAI_ENDPOINT"
echo " Search endpoint : $SEARCH_ENDPOINT"
echo " .env written to : ${PROJECT_ROOT}/.env"
echo ""
echo " Next steps:"
echo "   python scripts/download_pdfs.py"
echo "   python scripts/run_ingestion.py --all"
echo "   python main.py --interactive"
echo ""
echo " To tear down all resources:"
echo "   bash infra/teardown.sh"
echo "============================================================"
