import { useState, useEffect, useRef } from 'react'
import { LiveWaveform } from './components/ui/live-waveform'
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
  engine?: string
}

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [currentText, setCurrentText] = useState('')
  const [finalTexts, setFinalTexts] = useState<string[]>([])
  const [wsPort, setWsPort] = useState('8765')
  const [connecting, setConnecting] = useState(false)
  const [engine, setEngine] = useState<'kyutai' | 'parakeet'>('kyutai')
  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const isListeningRef = useRef(false)

  const startAudio = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      
      streamRef.current = stream
      audioContextRef.current = new AudioContext()
      
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream)
      
      const analyser = audioContextRef.current.createAnalyser()
      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.8
      
      processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1)
      
      processorRef.current.onaudioprocess = (e) => {
        if (wsRef.current?.readyState === WebSocket.OPEN && isListeningRef.current) {
          const inputSamples = e.inputBuffer.getChannelData(0)
          const inputSampleRate = audioContextRef.current!.sampleRate
          const targetSampleRate = 24000
          
          const resampled = resample(inputSamples, inputSampleRate, targetSampleRate)
          wsRef.current.send(resampled.buffer)
        }
      }
      
      sourceRef.current.connect(analyser)
      analyser.connect(processorRef.current)
      processorRef.current.connect(audioContextRef.current.destination)
      
      analyserRef.current = analyser
      isListeningRef.current = true
      setIsListening(true)
    } catch (error) {
      console.error('Failed to start audio:', error)
      throw error
    }
  }
  
  const resample = (samples: Float32Array, fromRate: number, toRate: number): Float32Array => {
    if (fromRate === toRate) {
      return samples
    }
    
    const ratio = fromRate / toRate
    const newLength = Math.round(samples.length / ratio)
    const result = new Float32Array(newLength)
    
    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio
      const srcIndexFloor = Math.floor(srcIndex)
      const srcIndexCeil = Math.min(srcIndexFloor + 1, samples.length - 1)
      const t = srcIndex - srcIndexFloor
      
      result[i] = samples[srcIndexFloor] * (1 - t) + samples[srcIndexCeil] * t
    }
    
    return result
  }

  const stopAudio = () => {
    isListeningRef.current = false
    setIsListening(false)
    
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect()
      analyserRef.current = null
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
  }

  const connect = async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setConnecting(true)
    const ws = new WebSocket(`ws://localhost:${wsPort}/`)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      setConnecting(false)
      ws.send(JSON.stringify({ type: 'setengine', engine }))
    }

    ws.onclose = () => {
      setIsConnected(false)
      setConnecting(false)
      stopAudio()
    }

    ws.onerror = () => {
      setConnecting(false)
    }

    ws.onmessage = (event) => {
      const message: TranscriptionMessage = JSON.parse(event.data)

      switch (message.type) {
        case 'word':
          setCurrentText(prev => prev + (message.word || '') + ' ')
          break
        case 'final':
          if (message.text) {
            setFinalTexts(prev => [...prev, message.text!])
            setCurrentText('')
          }
          break
        case 'enginechanged':
          if (message.engine) {
            setEngine(message.engine as 'kyutai' | 'parakeet')
          }
          break
      }
    }
  }

  const disconnect = () => {
    stopAudio()
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
      setIsConnected(false)
    }
  }

  const sendCommand = (command: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(command))
    }
  }

  const handleEngineChange = (value: 'kyutai' | 'parakeet') => {
    setEngine(value)
    sendCommand({ type: 'setengine', engine: value })
  }

  const handleToggleTranscription = async () => {
    if (!isListening) {
      try {
        await startAudio()
      } catch (error) {
        console.error('Failed to start listening:', error)
      }
    } else {
      stopAudio()
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
                placeholder="Port (default: 8765)"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isConnected}
              />
              <select
                value={engine}
                onChange={(e) => handleEngineChange(e.target.value as 'kyutai' | 'parakeet')}
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!isConnected}
              >
                <option value="kyutai">Kyutai</option>
                <option value="parakeet">Parakeet</option>
              </select>
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
            <LiveWaveform
              active={isListening}
              processing={false}
              height={100}
              barWidth={3}
              barGap={2}
              mode="static"
              sensitivity={1.5}
              analyser={analyserRef.current}
            />
            <div className="flex flex-wrap gap-2">
              <Button
                onClick={handleToggleTranscription}
                disabled={!isConnected}
                variant={isListening ? 'default' : 'outline'}
              >
                {isListening ? (
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
