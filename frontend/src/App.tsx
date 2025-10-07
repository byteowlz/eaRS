import { useState, useEffect, useRef } from 'react'
import { TranscriptionWaveform } from './components/transcription-waveform'
import { Button } from './components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { ThemeToggle } from './components/theme-toggle'
import { Mic, MicOff, RotateCcw } from 'lucide-react'

interface WordData {
  word: string
  start_time: number
  end_time?: number
}

interface TranscriptionMessage {
  type: string
  word?: string
  start_time?: number
  end_time?: number
  timestamp?: number
  text?: string
  words?: WordData[]
  paused?: boolean
  lang?: string
}

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [isPaused, setIsPaused] = useState(true)
  const [currentText, setCurrentText] = useState('')
  const [finalTexts, setFinalTexts] = useState<string[]>([])
  const [wsPort, setWsPort] = useState('8080')
  const [connecting, setConnecting] = useState(false)
  const [hasActivity, setHasActivity] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const activityTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    
    setConnecting(true)
    const ws = new WebSocket(`ws://localhost:${wsPort}/`)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      setConnecting(false)
      setIsPaused(true)
    }

    ws.onclose = () => {
      setIsConnected(false)
      setConnecting(false)
      setIsPaused(true)
    }

    ws.onerror = () => {
      setConnecting(false)
    }

    ws.onmessage = (event) => {
      const message: TranscriptionMessage = JSON.parse(event.data)

      switch (message.type) {
        case 'word':
          setCurrentText(prev => prev + (message.word || '') + ' ')
          setHasActivity(true)
          if (activityTimeoutRef.current) {
            clearTimeout(activityTimeoutRef.current)
          }
          activityTimeoutRef.current = setTimeout(() => {
            setHasActivity(false)
          }, 100)
          break
        case 'final':
          if (message.text) {
            setFinalTexts(prev => [...prev, message.text!])
            setCurrentText('')
          }
          setHasActivity(true)
          if (activityTimeoutRef.current) {
            clearTimeout(activityTimeoutRef.current)
          }
          activityTimeoutRef.current = setTimeout(() => {
            setHasActivity(false)
          }, 100)
          break
        case 'status':
          setIsPaused(message.paused ?? true)
          break
      }
    }
  }

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
      setIsConnected(false)
      setIsPaused(true)
    }
  }

  const sendCommand = (command: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(command))
    }
  }

  const handleToggleTranscription = () => {
    if (isPaused) {
      sendCommand({ type: 'resume' })
      setIsPaused(false)
    } else {
      sendCommand({ type: 'pause' })
      setIsPaused(true)
    }
  }

  const handleRestart = () => {
    sendCommand({ type: 'restart' })
    setCurrentText('')
    setFinalTexts([])
  }

  const handleClearTranscripts = () => {
    setCurrentText('')
    setFinalTexts([])
  }

  useEffect(() => {
    return () => {
      disconnect()
      if (activityTimeoutRef.current) {
        clearTimeout(activityTimeoutRef.current)
      }
    }
  }, [])

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-4xl space-y-6">
        <div className="flex items-center justify-between">
          <div className="text-center flex-1">
            <h1 className="text-4xl font-bold tracking-tight">eaRS Live Transcription</h1>
            <p className="text-muted-foreground mt-2">Real-time speech-to-text with live audio visualization</p>
          </div>
          <ThemeToggle />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Connection</CardTitle>
            <CardDescription>Connect to eaRS WebSocket server</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <input
                type="text"
                value={wsPort}
                onChange={(e) => setWsPort(e.target.value)}
                placeholder="Port (default: 8080)"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isConnected}
              />
              {!isConnected ? (
                <Button onClick={connect} disabled={connecting}>
                  {connecting ? 'Connecting...' : 'Connect'}
                </Button>
              ) : (
                <Button onClick={disconnect} variant="destructive">
                  Disconnect
                </Button>
              )}
            </div>
            <div className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm text-muted-foreground">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Audio Visualization</CardTitle>
            <CardDescription>Transcription activity visualization</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <TranscriptionWaveform
              active={isConnected && !isPaused}
              processing={isConnected && isPaused}
              activity={hasActivity}
              height={100}
              barWidth={3}
              barGap={2}
            />
            <div className="flex flex-wrap gap-2">
              <Button
                onClick={handleToggleTranscription}
                disabled={!isConnected}
                variant={!isPaused ? 'default' : 'outline'}
              >
                {!isPaused ? (
                  <>
                    <Mic className="mr-2 h-4 w-4" />
                    Stop Listening
                  </>
                ) : (
                  <>
                    <MicOff className="mr-2 h-4 w-4" />
                    Start Listening
                  </>
                )}
              </Button>
              <Button onClick={handleRestart} disabled={!isConnected} variant="outline">
                <RotateCcw className="mr-2 h-4 w-4" />
                Restart
              </Button>
              <Button onClick={handleClearTranscripts} variant="outline">
                Clear
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Live Transcription</CardTitle>
            <CardDescription>Current utterance being transcribed</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="min-h-[60px] rounded-md border border-border bg-muted/50 p-4">
              {currentText ? (
                <p className="text-lg">{currentText}</p>
              ) : (
                <p className="text-muted-foreground italic">Waiting for speech...</p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Transcript History</CardTitle>
            <CardDescription>Completed utterances</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="max-h-[400px] space-y-3 overflow-y-auto rounded-md border border-border bg-muted/50 p-4">
              {finalTexts.length > 0 ? (
                finalTexts.map((text, index) => (
                  <div key={index} className="rounded-md bg-background p-3 shadow-sm">
                    <p className="text-sm">{text}</p>
                  </div>
                ))
              ) : (
                <p className="text-muted-foreground italic">No completed transcriptions yet</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
