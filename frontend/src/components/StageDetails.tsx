import { CodeSquareIcon, RepoIcon } from '@primer/octicons-react'

import type {
  CandidateSearchResult,
  CriticResult,
  NodeName,
  RankingResult,
  RepositoryEvaluation,
  StageState,
  UserRequirements,
} from '../api/types'
import { EvaluationMetrics } from './EvaluationMetrics'

const TITLES: Record<NodeName, string> = {
  parse_requirements: 'Requirement Parser',
  search_candidates: 'Candidate Search',
  evaluate_repositories: 'Repository Evaluation',
  rank_repositories: 'Repository Ranking',
  critic: 'Critic Review',
  increment_retry: 'Retry Control',
}

function Pills({ values }: { values: string[] }) {
  if (!values.length) return <span className="muted">未指定</span>
  return <>{values.map((value) => <span className="pill" key={value}>{value}</span>)}</>
}

function RequirementsView({ value }: { value: UserRequirements }) {
  return (
    <div className="detail-stack">
      <div className="detail-summary">
        <span className="detail-label">Topic</span>
        <strong>{value.topic}</strong>
        <p>{value.learning_goal}</p>
      </div>
      <div className="requirement-grid">
        <div><span className="detail-label">Language</span><div><Pills values={value.language} /></div></div>
        <div><span className="detail-label">Technology</span><div><Pills values={value.technology_stack} /></div></div>
        <div><span className="detail-label">Difficulty</span><div><span className="pill">{value.difficulty}</span></div></div>
        <div><span className="detail-label">Time budget</span><strong>{value.time_budget_in_hours ? `${value.time_budget_in_hours} hours` : '不限'}</strong></div>
      </div>
      <div className="list-columns">
        <div><span className="detail-label">Must have</span><ul>{value.must_have.map((item) => <li key={item}>{item}</li>)}</ul></div>
        <div><span className="detail-label">Nice to have</span><ul>{value.nice_to_have.map((item) => <li key={item}>{item}</li>)}</ul></div>
      </div>
      {value.exclusions.length > 0 && (
        <div><span className="detail-label">Exclusions</span><div><Pills values={value.exclusions} /></div></div>
      )}
    </div>
  )
}

function CandidatesView({ value }: { value: CandidateSearchResult }) {
  return (
    <div className="detail-stack">
      <div>
        <span className="detail-label">Search queries</span>
        <div className="query-list">
          {value.search_queries.map((query) => <code key={query}>{query}</code>)}
        </div>
      </div>
      <div className="candidate-list">
        {value.candidates.map((candidate) => (
          <article key={candidate.full_name}>
            <RepoIcon size={16} />
            <div><strong>{candidate.full_name}</strong><p>{candidate.reason}</p></div>
          </article>
        ))}
      </div>
    </div>
  )
}

function EvaluationsView({ values }: { values: RepositoryEvaluation[] }) {
  return (
    <div className="evaluation-list">
      {values.map((evaluation) => (
        <details key={evaluation.full_name}>
          <summary>
            <RepoIcon size={16} />
            <strong>{evaluation.full_name}</strong>
            <span>
              {evaluation.suitable_for_next_stage ? 'Eligible' : 'Filtered'} ·{' '}
              {evaluation.requirement_match}/100 match
            </span>
          </summary>
          <div className="evaluation-body">
            <p>{evaluation.purpose_summary}</p>
            <EvaluationMetrics evaluation={evaluation} />
            <div className="evaluation-columns">
              <div>
                <span className="detail-label">Must-have checks</span>
                <ul className="check-list">
                  {evaluation.must_have_checks.map((check) => (
                    <li key={check.requirement}>
                      <span className={`check-status check-${check.status}`}>{check.status}</span>
                      <strong>{check.requirement}</strong>
                      <p>{check.evidence}</p>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="evaluation-notes">
                <div><span className="detail-label">Strengths</span><ul>{evaluation.strengths.map((item) => <li key={item}>{item}</li>)}</ul></div>
                <div><span className="detail-label">Weaknesses</span><ul>{evaluation.weaknesses.map((item) => <li key={item}>{item}</li>)}</ul></div>
                {evaluation.uncertainties.length > 0 && <div><span className="detail-label">Uncertainties</span><ul>{evaluation.uncertainties.map((item) => <li key={item}>{item}</li>)}</ul></div>}
              </div>
            </div>
          </div>
        </details>
      ))}
    </div>
  )
}

function RankingView({ value }: { value: RankingResult }) {
  return (
    <div className="detail-stack">
      <div className="detail-summary">
        <span className="detail-label">Best repository</span>
        <strong>{value.best_repository}</strong>
        <p>{value.final_reason}</p>
      </div>
      <div className="candidate-list">
        {[...value.top_repositories]
          .sort((left, right) => left.final_rank - right.final_rank)
          .map((repository) => (
            <article key={repository.full_name}>
              <span className="compact-rank">#{repository.final_rank}</span>
              <div>
                <strong>{repository.full_name}</strong>
                <p>{repository.recommendation_reason}</p>
                <span className="compact-score">Score {repository.base_score.toFixed(2)}</span>
              </div>
            </article>
          ))}
      </div>
    </div>
  )
}

function CriticView({ value }: { value: CriticResult }) {
  return (
    <div className="detail-stack">
      <div className="critic-stage-summary">
        <span className={`check-status ${value.passed ? 'check-satisfied' : 'check-not_satisfied'}`}>
          {value.passed ? 'passed' : 'needs attention'}
        </span>
        <strong>{value.confidence}% confidence</strong>
        <p>{value.feedback}</p>
      </div>
      {value.issues.length > 0 ? (
        <div className="issue-list">
          {value.issues.map((issue, index) => (
            <article key={`${issue.category}-${index}`}>
              <span className={`check-status check-${issue.severity === 'critical' ? 'not_satisfied' : 'uncertain'}`}>
                {issue.severity}
              </span>
              <div><strong>{issue.category}</strong><p>{issue.description}</p></div>
            </article>
          ))}
        </div>
      ) : (
        <p className="no-issues">没有发现证据、硬性要求或排序一致性问题。</p>
      )}
    </div>
  )
}

export function StageDetails({ stage }: { stage: StageState }) {
  const output = stage.output

  return (
    <section className="detail-panel" aria-live="polite">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">Stage output</span>
          <h2>{TITLES[stage.name]}</h2>
        </div>
        <span className={`stage-badge status-${stage.status}`}>{stage.status}</span>
      </div>

      {stage.status === 'pending' && (
        <div className="empty-detail"><CodeSquareIcon size={24} /><p>等待上游节点完成。</p></div>
      )}
      {stage.status === 'running' && (
        <div className="empty-detail running-detail"><span className="pulse-ring" /><p>Agent 正在处理这个阶段…</p></div>
      )}
      {stage.status === 'failed' && (
        <div className="empty-detail failed-detail"><p>这个阶段未能完成，请查看页面顶部的错误信息。</p></div>
      )}
      {stage.status === 'completed' && output && (
        <>
          {'requirements' in output && <RequirementsView value={output.requirements as UserRequirements} />}
          {'candidates' in output && <CandidatesView value={output.candidates as CandidateSearchResult} />}
          {'evaluations' in output && <EvaluationsView values={output.evaluations as RepositoryEvaluation[]} />}
          {'ranking' in output && <RankingView value={output.ranking as RankingResult} />}
          {'critic_result' in output && <CriticView value={output.critic_result as CriticResult} />}
          <details className="raw-output">
            <summary>查看完整结构化数据</summary>
            <pre className="json-output">{JSON.stringify(output, null, 2)}</pre>
          </details>
        </>
      )}
    </section>
  )
}
