#!/usr/bin/env bash
# infra/teardown.sh
# Deletes ALL Azure resources for the Financial Analyst Agent.
# This stops all billing immediately.
#
# Usage:
#   bash infra/teardown.sh
#
# The resource group name is read from .env (AZURE_RESOURCE_GROUP).
# You will be prompted to confirm before deletion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE"
    echo "       Run infra/provision.sh first, or set AZURE_RESOURCE_GROUP manually."
    exit 1
fi

# Source the .env file to read resource names
set -a
source "$ENV_FILE"
set +a

RG="${AZURE_RESOURCE_GROUP:-}"
SUB_ID="${AZURE_SUBSCRIPTION_ID:-}"
SUB_NAME="${AZURE_SUBSCRIPTION_NAME:-}"

if [[ -z "$RG" ]]; then
    echo "ERROR: AZURE_RESOURCE_GROUP not set in .env"
    exit 1
fi

# Detect whether a dedicated subscription was created by provision.sh
CREATED_NEW_SUB=false
DEFAULT_SUB_ID=$(az account show --query "id" -o tsv 2>/dev/null || echo "")
if [[ -n "$SUB_ID" && "$SUB_ID" != "$DEFAULT_SUB_ID" ]]; then
    CREATED_NEW_SUB=true
fi

echo "============================================================"
echo " WARNING: This will permanently delete:"
echo "   Resource Group : $RG"
echo "   and ALL resources inside it (OpenAI, Search, Storage)"
if [[ "$CREATED_NEW_SUB" == "true" ]]; then
echo "   Subscription   : $SUB_NAME ($SUB_ID)"
echo "   NOTE: Subscription cancellation will be requested."
echo "   Azure cancels subscriptions asynchronously; billing"
echo "   stops once the cancellation is processed (~24h)."
fi
echo "============================================================"
read -p " Type the resource group name to confirm: " CONFIRM

if [[ "$CONFIRM" != "$RG" ]]; then
    echo "Confirmation did not match. Aborting."
    exit 1
fi

# ── Switch to the target subscription before deleting ─────────────────────────
if [[ -n "$SUB_ID" ]]; then
    az account set --subscription "$SUB_ID"
fi

# ── Delete the resource group ──────────────────────────────────────────────────
echo ""
echo "Deleting resource group $RG ..."
az group delete --name "$RG" --yes --no-wait

# ── Cancel the dedicated subscription (if one was created) ────────────────────
if [[ "$CREATED_NEW_SUB" == "true" ]]; then
    echo ""
    echo "Cancelling subscription: $SUB_NAME ($SUB_ID) ..."
    # Azure CLI subscription cancellation (requires billing permissions)
    az rest \
        --method POST \
        --url "https://management.azure.com/subscriptions/${SUB_ID}/providers/Microsoft.Subscription/cancel?api-version=2021-10-01" \
        --output none 2>/dev/null && \
        echo "Subscription cancellation requested." || \
        echo "Could not cancel subscription via CLI — cancel manually at:"
        echo "  https://portal.azure.com/#view/Microsoft_Azure_Billing/SubscriptionsBlade"
fi

echo ""
echo "Tear-down initiated."
echo "Resource group deletion completes in ~2-5 minutes."
echo ""
echo "Verify at:"
echo "  https://portal.azure.com/#view/HubsExtension/BrowseResourceGroups"
