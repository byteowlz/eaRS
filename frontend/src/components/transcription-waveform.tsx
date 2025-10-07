import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

interface TranscriptionWaveformProps {
  active: boolean
  processing: boolean
  activity: boolean
  className?: string
  height?: number
  barWidth?: number
  barGap?: number
  barColor?: string
}

export function TranscriptionWaveform({
  active,
  processing,
  activity,
  className,
  height = 100,
  barWidth = 3,
  barGap = 2,
  barColor,
}: TranscriptionWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>(0)
  const activityLevelRef = useRef<number>(0)
  const lastActivityRef = useRef<number>(0)

  useEffect(() => {
    if (activity) {
      lastActivityRef.current = Date.now()
      activityLevelRef.current = 1.0
    }
  }, [activity])

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const resizeObserver = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1

      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
      canvas.style.width = `${rect.width}px`
      canvas.style.height = `${rect.height}px`

      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.scale(dpr, dpr)
      }
    })

    resizeObserver.observe(container)
    return () => resizeObserver.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let time = 0
    const animate = () => {
      const rect = canvas.getBoundingClientRect()
      ctx.clearRect(0, 0, rect.width, rect.height)

      const step = barWidth + barGap
      const barCount = Math.floor(rect.width / step)
      const centerY = rect.height / 2
      const halfCount = Math.floor(barCount / 2)

      const timeSinceActivity = Date.now() - lastActivityRef.current
      if (timeSinceActivity < 3000) {
        activityLevelRef.current = Math.max(0.1, 1.0 - timeSinceActivity / 3000)
      } else {
        activityLevelRef.current = 0.1
      }

      const computedBarColor =
        barColor ||
        getComputedStyle(canvas).color ||
        '#000'

      if (active && !processing) {
        time += 0.05 * activityLevelRef.current

        for (let i = 0; i < barCount; i++) {
          const normalizedPosition = (i - halfCount) / halfCount
          const centerWeight = 1 - Math.abs(normalizedPosition) * 0.3

          const wave1 = Math.sin(time * 2 + normalizedPosition * 4) * 0.3
          const wave2 = Math.sin(time * 1.5 - normalizedPosition * 3) * 0.25
          const wave3 = Math.cos(time * 2.5 + normalizedPosition * 2) * 0.2
          
          const combinedWave = wave1 + wave2 + wave3
          const baseValue = 0.15 + combinedWave * activityLevelRef.current
          const value = baseValue * centerWeight

          const x = i * step
          const barHeight = Math.max(4, Math.abs(value) * rect.height * 0.85)
          const y = centerY - barHeight / 2

          const alpha = 0.3 + Math.abs(value) * 0.7
          ctx.fillStyle = computedBarColor
          ctx.globalAlpha = alpha

          ctx.beginPath()
          ctx.roundRect(x, y, barWidth, barHeight, 1.5)
          ctx.fill()
        }
      } else if (processing) {
        for (let i = 0; i < barCount; i++) {
          const normalizedPosition = (i - halfCount) / halfCount
          const centerWeight = 1 - Math.abs(normalizedPosition) * 0.4

          const value = 0.2 * centerWeight

          const x = i * step
          const barHeight = Math.max(4, Math.abs(value) * rect.height * 0.8)
          const y = centerY - barHeight / 2

          ctx.fillStyle = '#ef4444'
          ctx.globalAlpha = 0.6

          ctx.beginPath()
          ctx.roundRect(x, y, barWidth, barHeight, 1.5)
          ctx.fill()
        }
      }

      ctx.globalAlpha = 1
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [active, processing, barWidth, barGap, barColor])

  return (
    <div
      className={cn('relative h-full w-full', className)}
      ref={containerRef}
      style={{ height: `${height}px` }}
      aria-label={
        active
          ? 'Live transcription waveform'
          : processing
            ? 'Waiting for transcription'
            : 'Transcription idle'
      }
      role="img"
    >
      {!active && !processing && (
        <div className="border-muted-foreground/20 absolute top-1/2 right-0 left-0 -translate-y-1/2 border-t-2 border-dotted" />
      )}
      <canvas
        className="block h-full w-full"
        ref={canvasRef}
        aria-hidden="true"
      />
    </div>
  )
}
