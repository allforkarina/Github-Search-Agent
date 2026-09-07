import type { RunEvent, RunEventType } from './types'

const EVENT_TYPES: RunEventType[] = [
  'run.started',
  'node.started',
  'node.completed',
  'retry.started',
  'run.completed',
  'run.cancelled',
  'run.failed',
]

interface CreateRunResponse {
  run_id: string
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string }
    return payload.detail || `Request failed with status ${response.status}`
  } catch {
    return `Request failed with status ${response.status}`
  }
}

export async function createRun(userRequest: string): Promise<string> {
  const response = await fetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_request: userRequest }),
  })

  if (!response.ok) {
    throw new Error(await readError(response))
  }

  const payload = (await response.json()) as CreateRunResponse
  return payload.run_id
}

export async function cancelRun(runId: string): Promise<void> {
  const response = await fetch(`/api/runs/${runId}/cancel`, {
    method: 'POST',
  })

  if (!response.ok) {
    throw new Error(await readError(response))
  }
}

export function subscribeToRun(
  runId: string,
  onEvent: (event: RunEvent) => void,
  onConnectionError: () => void,
): EventSource {
  const source = new EventSource(`/api/runs/${runId}/events`)

  for (const type of EVENT_TYPES) {
    source.addEventListener(type, (message) => {
      const event = JSON.parse((message as MessageEvent).data) as RunEvent
      onEvent(event)
    })
  }

  source.onerror = onConnectionError
  return source
}
