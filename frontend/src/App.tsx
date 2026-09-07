import { AlertIcon, CheckCircleFillIcon, XCircleFillIcon } from '@primer/octicons-react'
import { BaseStyles, ThemeProvider } from '@primer/react'
import { useEffect, useMemo, useState } from 'react'

import type { NodeName } from './api/types'
import { AgentTimeline } from './components/AgentTimeline'
import { AppHeader } from './components/AppHeader'
import { ResultsView } from './components/ResultsView'
import { SearchComposer } from './components/SearchComposer'
import { StageDetails } from './components/StageDetails'
import { useAgentRun } from './hooks/useAgentRun'
import { useColorMode } from './hooks/useColorMode'

const STAGE_ORDER: NodeName[] = [
  'parse_requirements',
  'search_candidates',
  'evaluate_repositories',
  'rank_repositories',
  'critic',
]

export default function App() {
  const { colorMode, isDark, toggle } = useColorMode()
  const { status, stages, result, error, retryMessage, start, cancel } = useAgentRun()
  const [selectedStage, setSelectedStage] = useState<NodeName>('parse_requirements')
  const hasRun = status !== 'idle'

  const activeStage = useMemo(
    () => STAGE_ORDER.find((name) => stages[name].status === 'running'),
    [stages],
  )

  useEffect(() => {
    if (activeStage) setSelectedStage(activeStage)
  }, [activeStage])

  return (
    <ThemeProvider colorMode={colorMode}>
      <BaseStyles className="app-surface" data-ui-theme={isDark ? 'dark' : 'light'}>
        <AppHeader isDark={isDark} onToggleTheme={toggle} />
        <main>
          <SearchComposer status={status} onSubmit={start} onCancel={cancel} />

          {error && (
            <div className="flash flash-error" role="alert">
              <XCircleFillIcon size={18} />
              <div><strong>搜索未完成</strong><p>{error}</p></div>
            </div>
          )}

          {status === 'cancelled' && (
            <div className="flash flash-warning" role="status">
              <AlertIcon size={18} />
              <div><strong>搜索已停止</strong><p>当前节点已结束，没有继续执行后续步骤。</p></div>
            </div>
          )}

          {status === 'completed' && (
            <div className="flash flash-success" role="status">
              <CheckCircleFillIcon size={18} />
              <div><strong>搜索完成</strong><p>候选仓库已经完成评估、排序与 Critic 审查。</p></div>
            </div>
          )}

          {hasRun && (
            <section className="workspace-section">
              <AgentTimeline
                stages={stages}
                selected={selectedStage}
                retryMessage={retryMessage}
                onSelect={setSelectedStage}
              />
              <StageDetails stage={stages[selectedStage]} />
            </section>
          )}

          {result && <ResultsView result={result} />}
        </main>
        <footer>
          <span>RepoScouts</span>
          <span>Local-first GitHub repository intelligence</span>
        </footer>
      </BaseStyles>
    </ThemeProvider>
  )
}
