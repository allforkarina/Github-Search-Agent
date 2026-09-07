import type { RepositoryEvaluation } from '../api/types'

const METRICS: Array<[keyof RepositoryEvaluation, string]> = [
  ['requirement_match', 'Requirement match'],
  ['learning_value', 'Learning value'],
  ['code_clarity', 'Code clarity'],
  ['documentation_quality', 'Documentation'],
  ['reproduction_feasibility', 'Reproducibility'],
  ['maintenance_health', 'Maintenance'],
]

export function EvaluationMetrics({ evaluation }: { evaluation: RepositoryEvaluation }) {
  return (
    <div className="metrics-grid">
      {METRICS.map(([key, label]) => {
        const value = evaluation[key] as number
        return (
          <div className="metric" key={key}>
            <div className="metric-label">
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
            <div className="metric-track" aria-label={`${label}: ${value}/100`}>
              <span style={{ width: `${value}%` }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}
