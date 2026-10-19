# Repository Research RAG V1：实现与阅读指南

## 交付状态

代码已实现，掌握待完成。开发者已经手写采集、切分和初版数据模型；剩余实现按开发者最新授权由助手完成。
原理讲解按开发者要求延后。当前完成的是代码接入与离线验证，不是完整真实模型/业务质量验收。
实现和离线验收期间没有下载 Embedding 权重，没有调用真实 GitHub/DeepSeek 请求，没有重跑原有 10 条 baseline。
代码按开发者授权提交为 AI 实现检查点；逐文件阅读与掌握验证继续进行，不创建版本标签或 Release。

## 第一版边界

- 全部搜索候选按相同预算采集：固定 commit，最多 40 个文件、单文件 128 KiB、总计 2 MiB。
- 网络请求截止检查保持开发者选择的 100 秒；这是协作式检查，不是进程级强制终止。
- Python AST、Markdown ATX 标题、其他文本按整行窗口切分；源码不执行。
- 本地 multilingual-e5-small + BM25L，各召回 20 条；RRF 每需求保留最多 5 条。
- 最多选择 20 个证据片段，片段序列化合计预算 24000 字符；Evaluator 变量输入合计最多 48000 字符。
- 模型固定 revision：614241f622f53c4eeff9890bdc4f31cfecc418b3。
- 默认 local_files_only=True；模块导入和用户请求都不会自动下载缺失模型。
- 未有权重时：BM25 + 字节预算切分，明确记录 local_embedding_unavailable / byte_budget_chunking。
- 同一次 API 请求固定快照，Critic 重试复用索引。请求结束、取消或异常后释放仓库内存缓存。
- 不实现 Cross-Encoder、跨请求持久向量库、跨 commit 文件增量更新、多语言 AST 和 LLM Query Rewrite。

## 数据流

```text
RunManager.research_scope()
  -> LangGraph.evaluate_repositories
  -> collect_repository_evidence(candidate, requirements)
  -> ResearchSession
       -> fixed commit/tree/blob ingestion
       -> independent requirement tasks
       -> chunking + local model or explicit byte-budget fallback
       -> RequestIndexCache -> Dense + BM25L -> RRF
       -> bounded, redacted Evidence pack
  -> Evaluator (existing cloud LLM)
  -> ground_evaluation(): validate IDs, version, repository and task coverage
  -> Ranker -> Critic -> optional evaluator retry (same ResearchSession)
```

本地 Embedding 不改变现有 Evaluator 仍调用云端 DeepSeek 的事实。发送给 Evaluator 的是预算内、
已脱敏的检索片段；完整文件、索引和向量留在进程内存，不进入 Graph state / API events / Trace stage summaries。
README 和源码使用同一 commit 的检索片段。GitHub stars 等 metadata 是请求时数据，不作为版本化源码证据。

## 文件清单与建议阅读顺序

表中“完善”表示在开发者已经完成的代码上修改；“保留”表示本轮没有替换该业务文件。

| 顺序 | 文件 | 本轮工作 / 主要职责 |
| --- | --- | --- |
| 1 | [rag/config.py](../../reposcouts/rag/config.py) | 完善：固定模型版本、本地缓存模式、批量大小、证据和输入预算 |
| 2 | [rag/schemas.py](../../reposcouts/rag/schemas.py) | 完善：检索结果、任务、引用、证据包模型 |
| 3 | [rag/github_source.py](../../reposcouts/rag/github_source.py) | 完善：原有固定版本读取，增加原生 Tool Trace 和敏感路径过滤 |
| 4 | [rag/chunking.py](../../reposcouts/rag/chunking.py) | 保留开发者实现：AST/Markdown/整行窗口切分 |
| 5 | [rag/embeddings.py](../../reposcouts/rag/embeddings.py) | 新增：延迟加载、本地文件模式、真实 token 计数、E5 前缀 |
| 6 | [rag/retrieval.py](../../reposcouts/rag/retrieval.py) | 新增：中英文标识符分词、BM25L、Dense、RRF、降级 |
| 7 | [rag/index_cache.py](../../reposcouts/rag/index_cache.py) | 新增：模型/版本/配置/源码感知的内存索引缓存 |
| 8 | [rag/requirements.py](../../reposcouts/rag/requirements.py) | 新增：拆显式合取，保留 OR/否定/C++ 等原意与父需求 |
| 9 | [rag/security.py](../../reposcouts/rag/security.py) | 新增：复用统一脱敏规则，保留原始行位置 |
| 10 | [rag/evidence.py](../../reposcouts/rag/evidence.py) | 新增：轮流选择各需求证据、预算、整文件脱敏、固定版本引用 |
| 11 | [rag/grounding.py](../../reposcouts/rag/grounding.py) | 新增：校验证据 ID、来源和子任务覆盖；补缺失判断、约束分数 |
| 12 | [rag/service.py](../../reposcouts/rag/service.py) | 新增：串联上述组件，管理请求内快照/切分/索引生命周期 |
| 13 | [rag/tracing.py](../../reposcouts/rag/tracing.py) | 新增：通过现有 callbacks 记录摘要，实际对象由闭包保存 |
| 14 | [rag/evaluation.py](../../reposcouts/rag/evaluation.py) | 新增：Recall/MRR/nDCG/Hit Rate、判断 F1、人工支持度 Citation Precision |
| 15 | [rag/download_model.py](../../reposcouts/rag/download_model.py) | 新增：以后手动运行的显式下载入口；本轮未运行 |
| 16 | [core/schemas.py](../../reposcouts/core/schemas.py) | 修改：Evidence 增加 research；判断增加 evidence_ids/citations；评估增加摘要 |
| 17 | [github/repository_collector.py](../../reposcouts/github/repository_collector.py) | 修改：调用 ResearchSession，替代旧 README/目录拼接流程 |
| 18 | [stages/repository_evaluator.py](../../reposcouts/stages/repository_evaluator.py) | 修改：发送脱敏证据、接收 Critic 反馈、校验引用、隔离候选失败 |
| 19 | [stages/requirement_parser.py](../../reposcouts/stages/requirement_parser.py) | 修改：提示独立硬要求、保留 alternatives 和 qualifiers |
| 20 | [stages/ranker.py](../../reposcouts/stages/ranker.py) | 修改：空候选拒绝强推，验证最终推荐来自合法候选 |
| 21 | [stages/critic.py](../../reposcouts/stages/critic.py) | 修改：允许合法的空推荐结构 |
| 22 | [workflow/nodes.py](../../reposcouts/workflow/nodes.py) | 修改：传递 Critic 反馈、空推荐正常终止 |
| 23 | [api/run_manager.py](../../reposcouts/api/run_manager.py) | 修改：请求内研究作用域，返回 recommendation_status |
| 24 | [observability/tracing.py](../../reposcouts/observability/tracing.py) | 修改：统一遮盖明文 key/token/私钥，保留秘密跨行时的位置 |
| 25 | [observability/metrics.py](../../reposcouts/observability/metrics.py) | 修改：区分 insufficient_evidence，避免把空推荐当作 passed |

`requirements.txt` 中的 RAG 依赖是开发者此前添加，本轮保留。前端和 Graph 拓扑保持原样；
现有 evidence 文本中包含可复制的来源地址，结构化 citations 同时保存在 API 结果中。

## 判断规则与局限

- satisfied / not_satisfied 必须引用该父需求下检索任务允许的 evidence_ids。
- 不存在、跨仓库、跨 SHA、缺失或没有覆盖全部子任务的引用会降为 uncertain。
- 检索不到不等于 not_satisfied；需要明确反证才能判不满足，这条语义要求由 Evaluator 提示约束。
- 程序生成 GitHub blob/SHA/path/#Lx-Ly 地址，不接受模型生成的引用地址。
- requirement_match 按已满足硬要求比例重算；有硬要求未核实时不进入最终推荐。
- 引用校验是来源和结构校验，不自动证明片段语义支持结论。实现完整程度、依赖声明与真实功能的区分，
  仍需真实模型验收和人工评测。仓库内的提示注入被标为不可信数据，但提示词本身不能保证彻底消除此风险。
- private key 在整文件层面先脱敏再截取引用，避免跨 chunk 泄露；过滤文件/正则脱敏也不能保证识别所有秘密格式。
- 无权重时的字节预算只是 BM25 降级路径；不当作真实 tokenizer，也不声称 Dense 已运行。
- 所有候选证据不足时 run.completed 可以正常结束，recommendation_status 和摘要 outcome 都是 insufficient_evidence。
  此时 Critic passed 表示“没有作出无依据的推荐”，不表示找到了符合要求的仓库。

## 观测与数据保留

使用 RunManager 创建的显式 tracer，子阶段继承当前 LangGraph Node 的 callbacks。
新增 ingestion、GitHub Tool、任务分解、切分、索引、embedding、dense、bm25、rrf、evidence、judgment spans。
新增阶段的 invoke 输入/输出只包含计数、ID、模式、版本和受控错误码，不包含完整源码和向量。
现有 LLM Trace 仍遵循原 client 的脱敏/截断规则。没有额外创建持久源码日志或跨请求仓库缓存。
权重是进程级缓存；仓库数据是请求级缓存，两者的生命周期不同。

## 测试与评测文件

- [test_rag_source_chunking.py](../../tests/test_rag_source_chunking.py)：此前整理的 25 项采集/切分测试，继续沿用。
- [test_rag_retrieval.py](../../tests/test_rag_retrieval.py)：本轮新增，真实 BM25 + 可控向量验证检索/缓存。
- [test_rag_integration.py](../../tests/test_rag_integration.py)：本轮新增，证据、脱敏、来源、实际 Graph 重试和 Trace 层级。
- [benchmark_v1.json](../../tests/rag/benchmark_v1.json)：本轮新增的 5 条合成标注样例，标签由助手编写，等待人工审阅。
- [run_rag_benchmark.py](../../tests/run_rag_benchmark.py)：本轮新增的本地评测入口，默认 BM25，按文件单位计分。

离线 BM25 sanity benchmark 已运行；这不代表真实 GitHub 语料上的语义检索质量，也不代表超过旧版 baseline。
最终 94 项测试通过；回归执行期间阻断外部网络，只保留 Windows 事件循环需要的本机通信。
Python 源文件语法解析与 git diff --check 通过。
原来的 10 条真实业务 baseline 没有改变。人工标注、真实 Hybrid 质量、真实 GitHub + DeepSeek + LangSmith 请求尚待后续。
宏 F1 对 gold/prediction 中出现的类别取平均；Citation Precision 必须输入独立人工支持标签，不能拿 URL 有效性替代。

## 当前环境与执行方式

当前实际环境：Python 3.12.13；torch 2.9.1+cu126；sentence-transformers 5.7.0；
transformers 5.20.1；rank-bm25 0.2.2；numpy 2.5.2。pip check 通过，CUDA 可用。
已在禁网条件下只导入模型库并检查接口参数，没有实例化或下载模型权重。

```powershell
# 当前可运行：不需要模型权重，不调用外部业务 API
python -m unittest discover -s tests -p "test_*.py"
python tests/run_rag_benchmark.py --mode bm25 --output tests/artifacts/rag-benchmark-v1/bm25.json
```

以后决定下载权重时再运行以下命令（本轮未运行）：

```powershell
python -m reposcouts.rag.download_model
python tests/run_rag_benchmark.py --mode hybrid --output tests/artifacts/rag-benchmark-v1/hybrid.json
```

Hybrid 验收只允许本地缓存权重；任何查询实际降级 BM25 时命令会报验证失败。
下载与 Hybrid 检查之后，再进行一次真实请求的源码引用和 LangSmith 验收，最后补独立人工标注集、与原 10 条需求对比。

## 后续共同阅读

先按表格 1–15 阅读模块，再沿 collector → Evaluator → RunManager 串联调用链。
重点重新解释固定 SHA、整行引用、RRF 的排名融合、模型/仓库缓存生命周期和来源校验与语义支持的区别。
按开发者要求，不增加掌握问答门禁；掌握验证延期，代码检查点不代表已完成真实模型验收或正式发布。
