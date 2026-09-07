# RepoScouts Web Interface

## 本次解决的问题

将原本写死在 `main.py` 中并输出到终端的 Agent 调用改为本地 Web 应用。用户从浏览器提交自然语言需求，在网页中查看工作流节点状态、结构化中间结果、最终排名和 Critic 审查结果。

## 采用的方案

- FastAPI 提供任务创建、SSE 事件流、取消和健康检查接口。
- React、TypeScript 和 Vite 构建单页界面。
- GitHub Primer 与 Octicons 提供 GitHub 风格的视觉语言。
- LangGraph 使用 `stream_mode="updates"` 复用现有图并暴露节点完成事件。
- 一个线程执行一个同步工作流；本地应用同一时间只允许一个活动任务。
- 取消使用节点间协作式停止，不强制终止正在执行的外部请求。
- 不使用数据库，不保存搜索历史。

## 修改的模块

- `main.py`：本地 Web 服务启动入口。
- `reposcouts/api/`：HTTP API、序列化和临时任务管理。
- `frontend/`：React Web 应用。
- `tests/`：API、状态流、取消和浏览器冒烟测试。

## 关键数据流与状态

```text
Browser POST user_request
  -> RunManager creates transient run
  -> LangGraph streams node updates
  -> RunManager serializes Pydantic outputs
  -> SSE sends events to browser
  -> React renders timeline and final ranking
```

任务状态为 `running -> completed | cancelled | failed`，取消请求会先进入 `cancelling`。浏览器刷新后不会恢复任务视图。

## 主要技术决策

- SSE 足以支持服务器到浏览器的单向进度推送，比 WebSocket 更符合当前数据流。
- API Key 和 GitHub Token 只由 Python 进程读取，浏览器接口不接受或返回密钥。
- React 类型与 Pydantic 输出字段保持一一对应。
- UI 默认跟随系统主题，允许本次页面会话内手动切换。

## 开发者主动修改

尚未完成。建议开发者亲手为 `RepositoryEvaluation.suitable_for_next_stage` 在评估详情中增加状态标识，并重新运行前端检查。

## 测试结果

- Python `unittest`：API、SSE 完整运行、并发拒绝和取消路径通过。
- ESLint：通过。
- TypeScript 与 Vite 生产构建：通过。
- Playwright Chromium：初始页、主题切换、模拟 SSE 完整流程、结果页和移动端冒烟测试通过。

## 掌握状态

代码已实现，开发者掌握验证待完成。

## 已知限制

- 工作流节点内部的 GitHub/LLM 请求不会被强行终止。
- 当前任务存在于进程内存，不支持刷新恢复或历史查询。
- Primer React 使单页 JavaScript bundle 超过 Vite 默认的 500 KB 提示阈值，但 gzip 后约 103 KB，当前本地应用可接受。
- Python 环境的 `usercustomize` 仍可能报告 `mingw64ccompiler` 缺失，与本功能无关。

## 开放问题

- 尚未使用真实 DeepSeek/GitHub 请求执行完整 Web 搜索，以避免在自动测试中消耗外部额度。

## 后续改进

- 如果需要跨刷新恢复，再引入 SQLite 或 LangGraph checkpointer。
- 如果仓库元数据进入最终 Schema，可在结果卡片增加 Stars、Forks、License 和更新时间。
