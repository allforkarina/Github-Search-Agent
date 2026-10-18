# baseline-v1-20261019-023918

状态：completed。每条需求仅执行一次；失败结果保留，不补跑。

|需求|执行状态 / outcome|秒|LLM|Tool|Token|USD|最佳仓库|
|---|---|---:|---:|---:|---:|---:|---|
|LangGraph 入门与短时复现|completed / passed|83.54|14|24|64066|0.013627122|eliharoun/langgraph-weather-agent|
|中文文档与本地离线 RAG|failed / failed|23.847|5|3|10189|0.002831988|—|
|混合检索、重排与 RAG 评估|completed / passed|78.258|14|24|75712|0.014775774|DataTalksClub/llm-zoomcamp|
|浏览器自动化 Agent|completed / passed|83.394|14|27|95597|0.018043038|camel-ai/owl|
|多 Agent 协作研究流程|completed / passed|54.708|11|15|40476|0.008375988|axon011/multi-agent-pipeline|
|仓库分析与代码理解 Agent|completed / passed|48.624|9|9|29094|0.006694596|muoimeo/evaluating-long-context-rag-and-recursive-reading-for-llm-repository-qa|
|FastAPI 后台任务工程|completed / passed|72.809|13|24|71302|0.014299062|iam-abbas/FastAPI-Production-Boilerplate|
|React TypeScript 本地看板|completed / passed|81.69|15|27|74882|0.014789448|prashantkoirala465/Vrit-Technologies-Frontend-Internship-Task-B|
|PyTorch 时间序列预测复现|completed / passed|90.157|16|30|88285|0.016884468|awslabs/gluonts|
|本地数据处理流水线|completed / passed|99.148|15|30|97594|0.019561326|edwinweber/dbt_duckdb_demo_public|

费用是 LangSmith 估算，未知为 null（表格显示 —），不按零费用处理。

Critic 通过是内部信号，独立质量评审尚未完成；十个不同输入各一次不能衡量同一输入的随机波动。

代码版本和逐文件哈希见 manifest.json；依赖版本见 dependencies.json 和 requirements-lock.txt。完整源码副本仅本地留存，本次提交包含对应实现。
GitHub 数据会变化；保留查询、候选、元数据和已脱敏工具输出的 SHA256，不能保证外部数据完全重放。

首次快照可能包含未提交代码；以 source_fingerprint 和每个文件的 original_sha256 标识实际版本。
这是精简归档：保留逐条业务结果、节点耗时、LLM/Tool 调用指标、证据指纹与实际 Critic 轨迹；不含原始 LLM 消息、重复源码副本、临时诊断文件或 .env。

## Trace 链接

- 01_langgraph_beginner: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15050-2123-7962-bfd0-5989bbd25521?poll=true
- 02_local_chinese_rag: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15051-d0d9-78e3-9a3e-2d53f9e252c1?poll=true
- 03_advanced_rag_evaluation: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15052-93cc-7893-9287-84feab92841e?poll=true
- 04_browser_automation_agent: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15054-2796-7693-94da-d621d5520313?poll=true
- 05_multi_agent_research: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15055-d787-7350-9e6e-94d71b2c7e9d?poll=true
- 06_repository_analysis_agent: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15057-190f-7c52-baa4-efed5093ab91?poll=true
- 07_fastapi_background_tasks: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15058-3b84-7343-910c-6a7d2ad242ca?poll=true
- 08_react_kanban: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a15059-c07a-7ea3-83dc-09f4b07dc9d0?poll=true
- 09_pytorch_time_series: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a1505b-64e0-70f2-95ed-f4b44dc5dd1e?poll=true
- 10_local_data_pipeline: https://smith.langchain.com/o/e050f023-2439-4a82-b806-2ae3633fb031/projects/p/e310a231-2e14-4aa2-a592-a5c147883f55/r/01a1505d-3137-7b63-89bb-62798fd329fe?poll=true

本轮未触发 Critic 重试，记录中的 retry_count 均为 0。平均耗时包含失败样本，比较性能时应同时检查执行状态。
