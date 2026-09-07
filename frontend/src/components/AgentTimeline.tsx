import {
  CheckCircleFillIcon,
  DotFillIcon,
  IssueOpenedIcon,
  SearchIcon,
  SyncIcon,
  XCircleFillIcon,
} from '@primer/octicons-react'

import type { NodeName, StageState } from '../api/types'

const STAGES: Array<{
  name: NodeName
  title: string
  caption: string
}> = [
  { name: 'parse_requirements', title: 'Requirement Parser', caption: '理解搜索目标' },
  { name: 'search_candidates', title: 'Candidate Search', caption: '生成查询并搜索 GitHub' },
  { name: 'evaluate_repositories', title: 'Repository Evaluation', caption: '收集证据并独立评估' },
  { name: 'rank_repositories', title: 'Repository Ranking', caption: '计算得分并生成建议' },
  { name: 'critic', title: 'Critic Review', caption: '检查证据与排序一致性' },
]

interface AgentTimelineProps {
  stages: Record<NodeName, StageState>
  selected: NodeName
  retryMessage: string | null
  onSelect: (name: NodeName) => void
}

function StageIcon({ stage }: { stage: StageState }) {
  if (stage.status === 'completed') return <CheckCircleFillIcon size={18} />
  if (stage.status === 'failed') return <XCircleFillIcon size={18} />
  if (stage.status === 'running') return <SyncIcon className="spin" size={18} />
  return <DotFillIcon size={20} />
}

export function AgentTimeline({
  stages,
  selected,
  retryMessage,
  onSelect,
}: AgentTimelineProps) {
  return (
    <aside className="timeline-panel" aria-label="Agent 工作流">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">Agent trace</span>
          <h2>工作流</h2>
        </div>
        <SearchIcon size={18} />
      </div>

      <ol className="timeline">
        {STAGES.map(({ name, title, caption }) => {
          const stage = stages[name]
          return (
            <li key={name} className={`timeline-item status-${stage.status}`}>
              <button
                type="button"
                className={selected === name ? 'selected' : ''}
                onClick={() => onSelect(name)}
              >
                <span className="timeline-icon">
                  <StageIcon stage={stage} />
                </span>
                <span className="timeline-copy">
                  <strong>{title}</strong>
                  <small>{caption}</small>
                </span>
                {stage.attempts > 1 && (
                  <span className="attempt-badge">×{stage.attempts}</span>
                )}
              </button>
            </li>
          )
        })}
      </ol>

      {retryMessage && (
        <div className="retry-note">
          <IssueOpenedIcon size={16} />
          <span>{retryMessage}</span>
        </div>
      )}
    </aside>
  )
}
