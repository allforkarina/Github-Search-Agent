import { MarkGithubIcon, MoonIcon, SunIcon } from '@primer/octicons-react'

interface AppHeaderProps {
  isDark: boolean
  onToggleTheme: () => void
}

export function AppHeader({ isDark, onToggleTheme }: AppHeaderProps) {
  return (
    <header className="app-header">
      <a className="brand" href="/" aria-label="RepoScouts 首页">
        <span className="brand-mark" aria-hidden="true">
          <MarkGithubIcon size={22} />
        </span>
        <span>RepoScouts</span>
        <span className="brand-edition">Agent</span>
      </a>

      <div className="header-actions">
        <span className="local-badge">
          <span className="local-dot" aria-hidden="true" />
          Local
        </span>
        <button
          className="icon-button"
          type="button"
          onClick={onToggleTheme}
          aria-label={isDark ? '切换到浅色模式' : '切换到深色模式'}
          title={isDark ? '切换到浅色模式' : '切换到深色模式'}
        >
          {isDark ? <SunIcon size={16} /> : <MoonIcon size={16} />}
        </button>
      </div>
    </header>
  )
}
