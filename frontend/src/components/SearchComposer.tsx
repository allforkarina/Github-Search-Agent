import { PaperAirplaneIcon, StopIcon } from '@primer/octicons-react'
import { useState } from 'react'

import type { RunStatus } from '../api/types'

const EXAMPLES = [
  '寻找适合初学者、两小时内可以复现的 Python LangGraph Agent 项目，要求 README 清晰。',
  '找一个使用 TypeScript 构建的 RAG 应用，架构完整，适合作为生产项目参考。',
]

interface SearchComposerProps {
  status: RunStatus
  onSubmit: (request: string) => void
  onCancel: () => void
}

export function SearchComposer({
  status,
  onSubmit,
  onCancel,
}: SearchComposerProps) {
  const [request, setRequest] = useState('')
  const isBusy = ['starting', 'running', 'cancelling'].includes(status)
  const canSubmit = request.trim().length >= 3 && !isBusy

  function submit() {
    if (canSubmit) {
      onSubmit(request.trim())
    }
  }

  return (
    <section className="search-section" aria-labelledby="search-title">
      <div className="eyebrow">GitHub repository intelligence</div>
      <h1 id="search-title">
        Find the repository
        <span> worth your time.</span>
      </h1>
      <p className="hero-copy">
        用自然语言描述目标。RepoScouts 会搜索、取证、评估并审查候选仓库，最后给出可解释的排序。
      </p>

      <div className={`composer ${isBusy ? 'composer-busy' : ''}`}>
        <label htmlFor="repo-request">你希望找到什么样的项目？</label>
        <textarea
          id="repo-request"
          value={request}
          onChange={(event) => setRequest(event.target.value)}
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
              submit()
            }
          }}
          disabled={isBusy}
          maxLength={4000}
          rows={5}
          placeholder="例如：我想学习如何使用 LangGraph 构建 AI Agent，希望项目结构清晰、文档完整，并且能在两小时内复现……"
        />
        <div className="composer-footer">
          <span className="keyboard-hint">Ctrl ↵ 运行</span>
          {isBusy ? (
            <button
              className="button button-danger"
              type="button"
              onClick={onCancel}
              disabled={status === 'starting' || status === 'cancelling'}
            >
              <StopIcon size={16} />
              {status === 'starting'
                ? '正在启动…'
                : status === 'cancelling'
                  ? '正在停止…'
                  : '停止搜索'}
            </button>
          ) : (
            <button
              className="button button-primary"
              type="button"
              onClick={submit}
              disabled={!canSubmit}
            >
              Search repositories
              <PaperAirplaneIcon size={16} />
            </button>
          )}
        </div>
      </div>

      {!isBusy && status === 'idle' && (
        <div className="examples" aria-label="搜索示例">
          <span>Try an example</span>
          {EXAMPLES.map((example) => (
            <button key={example} type="button" onClick={() => setRequest(example)}>
              {example}
            </button>
          ))}
        </div>
      )}
    </section>
  )
}
