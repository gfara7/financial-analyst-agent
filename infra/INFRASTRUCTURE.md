# Financial Analyst Agent — Azure Infrastructure

Provisioned on: 2026-03-05
Subscription: `finance-agent-mvp` (`daf9bb0a-f39d-4336-a6f8-85fc444ef8e7`)
Resource Group: `rg-financial-agent`
Location: `swedencentral`
Resource Suffix: `4bc61f`

---

## Resources

### 1. Azure OpenAI

| Property | Value |
|---|---|
| Resource Name | `aoai-fin-agent-4bc61f` |
| Kind | `OpenAI` |
| SKU | `S0` (Standard) |
| Location | `swedencentral` |
| Endpoint | `https://swedencentral.api.cognitive.microsoft.com/` |

#### Model Deployments

| Deployment Name | Model | Model Version | SKU | Capacity (TPM) |
|---|---|---|---|---|
| `text-embedding-ada-002` | `text-embedding-ada-002` | `2` | Standard | 100,000 |
| `gpt-4o` | `gpt-4o` | `2024-08-06` | Standard | 30,000 |

**API Version used in .env:** `2024-08-01-preview`

---

### 2. Azure AI Search

| Property | Value |
|---|---|
| Resource Name | `srch-fin-agent-4bc61f` |
| SKU | `basic` |
| Location | `swedencentral` |
| Replica Count | 1 |
| Partition Count | 1 |
| Semantic Search | free tier |
| Endpoint | `https://srch-fin-agent-4bc61f.search.windows.net` |
| Index Name | `financial-docs` |

---

### 3. Azure Blob Storage

| Property | Value |
|---|---|
| Account Name | `stfinagent4bc61f` |
| SKU | `Standard_LRS` |
| Kind | `StorageV2` |
| Location | `swedencentral` |
| Access Tier | Hot |
| HTTPS Only | Yes |
| Minimum TLS | TLS 1.0 |
| Container | `source-pdfs` |

---

## Environment Variables

All credentials are written to `.env` at the project root. Keys:

```
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
AZURE_SUBSCRIPTION_ID
AZURE_SUBSCRIPTION_NAME
AZURE_RESOURCE_GROUP
AZURE_AOAI_NAME
AZURE_SEARCH_NAME
AZURE_STORAGE_NAME
```

---

## Estimated Cost

| Resource | Tier | Est. Daily Cost |
|---|---|---|
| Azure OpenAI | S0 (pay-per-token) | ~$1–3/day depending on usage |
| Azure AI Search | Basic | ~$0.74/day |
| Azure Storage | Standard_LRS | ~$0.01/day |
| **Total** | | **~$2–4/day** |

---

## Re-provisioning Notes

The suffix `4bc61f` is persisted in `infra/.suffix`. Re-running `provision.sh` will reuse existing resources and skip creation steps.

**Known issue resolved:** Azure OpenAI uses soft-delete by default with a 48-hour retention window. If `provision.sh` fails with `FlagMustBeSetForRestore`, run `bash infra/purge_aoai.sh` first to purge all soft-deleted accounts, then re-run provisioning.

The `purge_aoai.sh` script handles:
1. Deleting any active `aoai-fin-agent-*` accounts in the resource group
2. Purging all soft-deleted `aoai-fin-agent-*` accounts across all regions

---

## Teardown

```bash
bash infra/teardown.sh
```

---

## Next Steps After Provisioning

```bash
python scripts/download_pdfs.py
python scripts/run_ingestion.py --all
python main.py --interactive
```
