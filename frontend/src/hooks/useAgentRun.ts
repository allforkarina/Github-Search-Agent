import { useCallback, useEffect, useRef, useState } from 'react'

import { cancelRun, createRun, subscribeToRun } from '../api/client'
import type {
  NodeName,
  RunEvent,
  RunResult,
  RunStatus,
  StageState,
} from '../api/types'

const NODE_NAMES: NodeName[] = [
  'parse_requirements',
  'search_candidates',
  'evaluate_repositories',
  'rank_repositories',
  'critic',
  'increment_retry',
]

function createInitialStages(): Record<NodeName, StageState> {
  return Object.fromEntries(
    NODE_NAMES.map((name) => [
      name,
      { name, status: 'pending', attempts: 0, output: null },
    ]),
  ) as Record<NodeName, StageState>
}

function errorMessage(value: unknown): string {
  return value instanceof Error ? value.message : '发生了未知错误。'
}

export function useAgentRun() {
  const [status, setStatus] = useState<RunStatus>('idle')
  const [runId, setRunId] = useState<string | null>(null)
  const [stages, setStages] = useState(createInitialStages)
  const [result, setResult] = useState<RunResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [retryMessage, setRetryMessage] = useState<string | null>(null)
  const sourceRef = useRef<EventSource | null>(null)
  const terminalEventRef = useRef(false)

  const closeSource = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
  }, [])

  useEffect(() => closeSource, [closeSource])

  const handleEvent = useCallback(
    (event: RunEvent) => {
      if (event.type === 'run.started') {
        setStatus('running')
        return
      }

      if (event.type === 'node.started' && event.node) {
        setStages((current) => ({
          ...current,
          [event.node as NodeName]: {
            ...current[event.node as NodeName],
            status: 'running',
            attempts: current[event.node as NodeName].attempts + 1,
          },
        }))
        return
      }

      if (event.type === 'node.completed' && event.node) {
        const output = (event.data.output || {}) as Record<string, unknown>
        setStages((current) => ({
          ...current,
          [event.node as NodeName]: {
            ...current[event.node as NodeName],
            status: 'completed',
            output,
          },
        }))
        return
      }

      if (event.type === 'retry.started') {
        const attempt = Number(event.data.attempt || 1)
        const maximum = Number(event.data.max_retries || 0)
        setRetryMessage(`Critic 请求重新处理 · ${attempt}/${maximum}`)
        setStages((current) => ({
          ...current,
          increment_retry: {
            ...current.increment_retry,
            status: 'running',
            attempts: current.increment_retry.attempts + 1,
            output: event.data,
          },
        }))
        return
      }

      if (event.type === 'run.completed') {
        terminalEventRef.current = true
        setResult(event.data as unknown as RunResult)
        setStatus('completed')
        closeSource()
        return
      }

      if (event.type === 'run.cancelled') {
        terminalEventRef.current = true
        setStatus('cancelled')
        closeSource()
        return
      }

      if (event.type === 'run.failed') {
        terminalEventRef.current = true
        if (event.node) {
          setStages((current) => ({
            ...current,
            [event.node as NodeName]: {
              ...current[event.node as NodeName],
              status: 'failed',
            },
          }))
        }
        setError(String(event.data.message || 'Repository search failed.'))
        setStatus('failed')
        closeSource()
      }
    },
    [closeSource],
  )

  const start = useCallback(
    async (userRequest: string) => {
      closeSource()
      terminalEventRef.current = false
      setStatus('starting')
      setRunId(null)
      setStages(createInitialStages())
      setResult(null)
      setError(null)
      setRetryMessage(null)

      try {
        const nextRunId = await createRun(userRequest)
        setRunId(nextRunId)
        sourceRef.current = subscribeToRun(
          nextRunId,
          handleEvent,
          () => {
            if (!terminalEventRef.current) {
              setError('与本地 Agent 服务的事件连接已断开。')
              setStatus('failed')
            }
          },
        )
      } catch (reason) {
        setError(errorMessage(reason))
        setStatus('failed')
      }
    },
    [closeSource, handleEvent],
  )

  const cancel = useCallback(async () => {
    if (!runId || (status !== 'running' && status !== 'starting')) {
      return
    }

    setStatus('cancelling')
    try {
      await cancelRun(runId)
    } catch (reason) {
      setError(errorMessage(reason))
      setStatus('failed')
    }
  }, [runId, status])

  return {
    status,
    stages,
    result,
    error,
    retryMessage,
    start,
    cancel,
  }
}
