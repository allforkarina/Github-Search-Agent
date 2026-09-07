export type RunStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'cancelling'
  | 'completed'
  | 'cancelled'
  | 'failed'

export type StageStatus = 'pending' | 'running' | 'completed' | 'failed'

export type NodeName =
  | 'parse_requirements'
  | 'search_candidates'
  | 'evaluate_repositories'
  | 'rank_repositories'
  | 'critic'
  | 'increment_retry'

export type RunEventType =
  | 'run.started'
  | 'node.started'
  | 'node.completed'
  | 'retry.started'
  | 'run.completed'
  | 'run.cancelled'
  | 'run.failed'

export interface RunEvent {
  type: RunEventType
  run_id: string
  node: NodeName | null
  data: Record<string, unknown>
}

export interface UserRequirements {
  topic: string
  language: string[]
  technology_stack: string[]
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'any'
  must_have: string[]
  nice_to_have: string[]
  exclusions: string[]
  learning_goal: string
  time_budget_in_hours: number | null
}

export interface CandidateRepository {
  full_name: string
  reason: string
}

export interface CandidateSearchResult {
  search_queries: string[]
  candidates: CandidateRepository[]
}

export interface RequirementCheck {
  requirement: string
  status: 'satisfied' | 'not_satisfied' | 'uncertain'
  evidence: string
}

export interface RepositoryEvaluation {
  full_name: string
  purpose_summary: string
  must_have_checks: RequirementCheck[]
  requirement_match: number
  learning_value: number
  code_clarity: number
  documentation_quality: number
  reproduction_feasibility: number
  maintenance_health: number
  strengths: string[]
  weaknesses: string[]
  uncertainties: string[]
  suitable_for_next_stage: boolean
}

export interface RankedRepository {
  full_name: string
  base_score: number
  score_rank: number
  final_rank: number
  recommendation_reason: string
  tradeoffs: string[]
  learning_path: string[]
  ranking_override: boolean
  override_reason: string | null
}

export interface RankingResult {
  top_repositories: RankedRepository[]
  best_repository: string
  final_reason: string
}

export interface CriticIssue {
  severity: 'warning' | 'critical'
  category: 'hard_requirement' | 'evidence' | 'ranking' | 'unsupported_claim'
  repository: string | null
  description: string
}

export interface CriticResult {
  passed: boolean
  confidence: number
  issues: CriticIssue[]
  retry_target: 'none' | 'ranker' | 'evaluator'
  feedback: string
}

export interface RunResult {
  requirements: UserRequirements | null
  candidates: CandidateSearchResult | null
  evaluations: RepositoryEvaluation[]
  ranking: RankingResult | null
  critic_result: CriticResult | null
  retry_count: number
}

export interface StageState {
  name: NodeName
  status: StageStatus
  attempts: number
  output: Record<string, unknown> | null
}
