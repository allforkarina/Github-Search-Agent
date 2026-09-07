# RepoScouts

```
根据开发者的自然语言需求，自主搜索、分析和比较 GitHub Repository，并推荐最适合当前需求的开源项目。
```

RepoScouts receives a natural-language goal, searches GitHub, evaluates candidate
repositories, ranks the best matches, and asks a Critic node to verify the final
recommendation.

## Local web application

### 1. Configure environment variables

Create a `.env` file in the project root:

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
GITHUB_TOKEN=your_github_token
```

Secrets are read only by the Python backend and are never sent to the browser.

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
```

### 3. Start RepoScouts

```powershell
python main.py
```

Open `http://127.0.0.1:8000` in a browser.

For frontend development, run `npm run dev` inside `frontend/` while the Python
server is running. Vite proxies `/api` requests to `http://127.0.0.1:8000`.

### Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
cd frontend
npm run lint
npm run build
```

The optional browser smoke test requires `requirements-dev.txt` followed by
`python -m playwright install chromium`.

## Workflow

```text
Natural language request
        -> Requirement Parser
        -> Candidate Search Agent
        -> Repository Evaluator
        -> Repository Ranker
        -> Critic
        -> retry evaluator/ranker when required
        -> final recommendation
```

The browser receives workflow updates through Server-Sent Events. Only one local
run can be active at a time, and cancellation takes effect between workflow nodes.
Search history is not persisted.

## Modules

- `reposcouts/core/`: configuration and Pydantic schemas.
- `reposcouts/github/`: GitHub REST API tools and repository evidence collection.
- `reposcouts/stages/`: requirement parsing, search, evaluation, ranking, and critique.
- `reposcouts/workflow/`: LangGraph state, nodes, and graph definition.
- `reposcouts/api/`: FastAPI routes, transient run management, SSE, and cancellation.
- `frontend/`: React, TypeScript, Vite, and GitHub Primer-based interface.
