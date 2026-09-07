import { useEffect, useState } from 'react'

export type ColorMode = 'day' | 'night' | 'auto'

export function useColorMode() {
  const [colorMode, setColorMode] = useState<ColorMode>('auto')
  const [systemIsDark, setSystemIsDark] = useState(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches,
  )

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const update = () => setSystemIsDark(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  const isDark = colorMode === 'night' || (colorMode === 'auto' && systemIsDark)

  function toggle() {
    setColorMode(isDark ? 'day' : 'night')
  }

  return { colorMode, isDark, toggle }
}
