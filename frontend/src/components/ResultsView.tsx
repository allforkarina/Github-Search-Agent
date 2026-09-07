import { TrophyIcon } from '@primer/octicons-react'

import type { RunResult } from '../api/types'
import { CriticSummary } from './CriticSummary'
import { RepositoryCard } from './RepositoryCard'

export function ResultsView({ result }: { result: RunResult }) {
  if (!result.ranking) return null

  const evaluations = new Map(
    result.evaluations.map((evaluation) => [evaluation.full_name, evaluation]),
  )
  const repositories = [...result.ranking.top_repositories].sort(
    (left, right) => left.final_rank - right.final_rank,
  )

  return (
    <section className="results-section" aria-labelledby="results-title">
      <div className="results-heading">
        <div className="trophy"><TrophyIcon size={22} /></div>
        <div>
          <span className="panel-kicker">Final recommendation</span>
          <h2 id="results-title">Repositories ranked for your goal</h2>
          <p>{result.ranking.final_reason}</p>
        </div>
      </div>

      <div className="repository-results">
        {repositories.map((repository) => (
          <RepositoryCard
            key={repository.full_name}
            repository={repository}
            evaluation={evaluations.get(repository.full_name)}
            best={repository.full_name === result.ranking?.best_repository}
          />
        ))}
      </div>

      {result.critic_result && <CriticSummary critic={result.critic_result} />}
    </section>
  )
}
