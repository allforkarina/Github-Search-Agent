import { ArrowUpRightIcon, CheckCircleFillIcon, RepoIcon } from '@primer/octicons-react'

import type { RankedRepository, RepositoryEvaluation } from '../api/types'
import { EvaluationMetrics } from './EvaluationMetrics'

interface RepositoryCardProps {
  repository: RankedRepository
  evaluation?: RepositoryEvaluation
  best: boolean
}

function repositoryUrl(fullName: string) {
  return `https://github.com/${fullName.split('/').map(encodeURIComponent).join('/')}`
}

export function RepositoryCard({ repository, evaluation, best }: RepositoryCardProps) {
  return (
    <article className={`repository-card ${best ? 'best-repository' : ''}`}>
      <div className="rank-column">
        <span>Rank</span>
        <strong>{String(repository.final_rank).padStart(2, '0')}</strong>
      </div>

      <div className="repository-content">
        <div className="repository-heading">
          <div>
            {best && <span className="best-label"><CheckCircleFillIcon size={14} /> Best match</span>}
            <h3><RepoIcon size={18} /> {repository.full_name}</h3>
          </div>
          <a
            className="github-link"
            href={repositoryUrl(repository.full_name)}
            target="_blank"
            rel="noreferrer"
          >
            View on GitHub <ArrowUpRightIcon size={14} />
          </a>
        </div>

        <p className="recommendation-reason">{repository.recommendation_reason}</p>

        <div className="score-line">
          <span>Weighted score</span>
          <strong>{repository.base_score.toFixed(2)}</strong>
          {repository.ranking_override && <span className="override-label">Rank adjusted by evidence</span>}
        </div>

        {repository.ranking_override && repository.override_reason && (
          <p className="override-reason">{repository.override_reason}</p>
        )}

        {evaluation && <EvaluationMetrics evaluation={evaluation} />}

        <div className="repository-columns">
          <div>
            <h4>Tradeoffs</h4>
            <ul>{repository.tradeoffs.map((item) => <li key={item}>{item}</li>)}</ul>
          </div>
          <div>
            <h4>Learning path</h4>
            <ol>{repository.learning_path.map((item) => <li key={item}>{item}</li>)}</ol>
          </div>
        </div>
      </div>
    </article>
  )
}
