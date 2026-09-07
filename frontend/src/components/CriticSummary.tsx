import { CheckCircleFillIcon, ShieldCheckIcon, XCircleFillIcon } from '@primer/octicons-react'

import type { CriticResult } from '../api/types'

export function CriticSummary({ critic }: { critic: CriticResult }) {
  return (
    <aside className={`critic-summary ${critic.passed ? 'critic-pass' : 'critic-warning'}`}>
      <div className="critic-icon"><ShieldCheckIcon size={22} /></div>
      <div className="critic-copy">
        <span className="panel-kicker">Critic verdict</span>
        <h3>
          {critic.passed ? <CheckCircleFillIcon size={16} /> : <XCircleFillIcon size={16} />}
          {critic.passed ? 'Recommendation verified' : 'Review completed with issues'}
        </h3>
        <p>{critic.feedback || '排序与证据检查已经完成。'}</p>
        {critic.issues.length > 0 && (
          <ul>{critic.issues.map((issue, index) => <li key={`${issue.category}-${index}`}>{issue.description}</li>)}</ul>
        )}
      </div>
      <div className="confidence"><strong>{critic.confidence}</strong><span>confidence</span></div>
    </aside>
  )
}
