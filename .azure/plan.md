# Azure Frontend Plan

## 1. Request
- Build a Python frontend so users can chat directly with the existing financial analyst agent.
- Keep using existing Azure OpenAI + Azure AI Search + Blob APIs already configured in `.env`.
- Deploy the frontend to the cheapest practical Azure web hosting option.
- Support localhost testing.
- Deliver on a separate feature branch and open a PR.

## 2. Mode
- `MODIFY` existing repository.

## 3. Decisions
- Frontend framework: Flask (minimal Python web UI, low dependency overhead).
- Hosting target: Azure Container Apps (consumption, min replicas `0`, max replicas `1`).
- Container registry: Azure Container Registry Basic (required for container image hosting).
- Deployment style: Azure CLI scripts (`infra/validate_frontend.sh`, `infra/deploy_frontend.sh`).
- Reuse existing environment variables from `.env` for AOAI/Search/Storage configuration.

## 4. Architecture
- Browser -> Flask UI (`/`) -> POST `/api/ask` -> LangGraph agent invocation -> response markdown back to browser.
- Azure path:
  - ACR builds and stores image.
  - Container Apps runs web container.
  - Container App environment + Log Analytics workspace provide runtime and logs.

## 5. Work Items
- [x] Add Flask frontend package and UI assets.
- [x] Add localhost launcher for testing.
- [x] Add Dockerfile for cloud runtime.
- [x] Add Azure validation/deploy scripts for frontend.
- [x] Validate deployment prerequisites with Azure CLI.
- [x] Deploy frontend to Azure.
- [ ] Push feature branch and create PR.

## 6. Status
- `Deployed` (validation and deployment completed).

## 7. Validation Proof
- Timestamp: `2026-03-05 23:47:58 +01:00`
- Command: `C:\Program Files\Git\bin\bash.exe infra/validate_frontend.sh`
  - Result: `Validation successful.`
- Command: `C:\Program Files\Git\bin\bash.exe infra/deploy_frontend.sh`
  - Result: deployment complete, URL output:
    - `https://financial-agent-web.yellowforest-0072335e.swedencentral.azurecontainerapps.io`
- Command: `Invoke-WebRequest https://financial-agent-web.yellowforest-0072335e.swedencentral.azurecontainerapps.io/healthz`
  - Result: `{"status":"ok"}`
- Command: POST `/api/ask` with query `"What are key credit risk factors for large US banks?"`
  - Result: 200 OK with generated report payload.
