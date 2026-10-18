# RepoScouts baseline v1

固定输入为 `cases_v1.json`：6 条 Agent/RAG、4 条其他开发任务；每条运行一次，按文件顺序串行执行。修改项目后应复用同一份数据集，新建一批结果，不覆盖旧目录。

在项目根目录使用 `llm-agent` 环境：

```powershell
python -B tests/run_baseline.py --prepare
python -B tests/run_baseline.py --execute tests/artifacts/baselines/<输出的批次目录>
```

`--prepare` 仅冻结当前代码、全部已安装包版本、固定输入与非敏感配置。它保存 Git commit、未提交状态和源码内容哈希，因此即使工作树未提交，也能定位当前实现。源码副本经过脱敏；若被改变，会分别记录原始哈希、副本哈希和标记。不会保存 `.env`。

`--execute` 会真实调用配置的模型、GitHub、LangSmith，并使用独立的 `reposcouts-baseline-v1` 项目。Agent 内部原有 Critic 重试保留；脚本不重跑失败需求。每个需求运行在独立进程，最多 15 分钟，超时会保留部分结果。运行中不要修改代码或环境；检测到变化时停止批次。

每条结果包含业务 run_id、最终和部分结果、逐节点耗时、LLM/Tool 次数、token 覆盖率、费用、Critic/重试轨迹、Trace 链接与脱敏检查。Token 和费用只累计叶子 LLM 调用；任意调用缺失数据时，相应合计为 null，另保留覆盖率。费用为 LangSmith 估算，不等于账单。汇总只在全部值已知时给出总费用。

保留 GitHub 查询、候选、仓库元数据和已脱敏工具输出指纹，不保存完整 LLM 提示词、模型原始回复或 README 全文。结果字符串最多 4000 字符。指纹针对上传并脱敏后的工具输入输出，无法保证 GitHub 数据的离线完全重放。

结果位于忽略 Git 的 `tests/artifacts/baselines/<批次>/`，其中 `BASELINE.md` 便于阅读、`summary.json` 便于后续自动对比，逐条 JSON 在 `cases/`。保留此目录作为冻结基线，未来运行新建目录。

Critic 通过率、硬要求检查和结构检查是辅助信号，不是独立质量结论。固定不同输入各一次能做配对比较，不能衡量同一输入的随机波动。后续有改进信号时，再针对同一输入重复运行，并做人工或独立评审。`review_focus` 是每条需求的人工评审重点，目前默认待评审。
