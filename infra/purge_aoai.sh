#!/usr/bin/env bash
# infra/purge_aoai.sh
# Deletes all active aoai-fin-agent-* accounts and purges their soft-deleted copies.
# Run this to reclaim quota before re-running provision.sh.
#
# Usage: bash infra/purge_aoai.sh

set -uo pipefail

RG="rg-financial-agent"

echo "=== Step 1: Delete active AOAI accounts in $RG ==="
if az group show --name "$RG" --output none 2>/dev/null; then
    ACTIVE=$(az cognitiveservices account list \
        --resource-group "$RG" \
        --query "[?starts_with(name, 'aoai-fin-agent')].name" \
        -o tsv 2>/dev/null || echo "")

    if [[ -z "$ACTIVE" ]]; then
        echo "  No active accounts found."
    else
        echo "$ACTIVE" | while read -r name; do
            echo "  Deleting: $name"
            az cognitiveservices account delete \
                --name "$name" \
                --resource-group "$RG" \
                --output none 2>/dev/null || true
        done
    fi
else
    echo "  Resource group $RG not found — skipping active deletion."
fi

echo ""
echo "=== Step 2: Purge all soft-deleted aoai-fin-agent-* accounts ==="
DELETED=$(az cognitiveservices account list-deleted \
    --query "[?starts_with(name, 'aoai-fin-agent')].{name:name, location:location, rg:resourceGroup}" \
    -o tsv 2>/dev/null || echo "")

if [[ -z "$DELETED" ]]; then
    echo "  No soft-deleted accounts found."
else
    echo "$DELETED" | while IFS=$'\t' read -r name location rg; do
        echo "  Purging: $name ($location)"
        az cognitiveservices account purge \
            --name "$name" \
            --location "$location" \
            --resource-group "$rg" \
            --output none 2>/dev/null && echo "    Purged." || echo "    Skipped (may already be purged)."
    done
fi

echo ""
echo "=== Done ==="
echo "Verify with: az cognitiveservices account list-deleted --output table"
