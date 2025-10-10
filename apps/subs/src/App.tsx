import { useState, useEffect, useRef } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { Settings, Mic, MicOff, RefreshCw } from 'lucide-react'
import { Button } from './components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './components/ui/select'
import { Label } from './components/ui/label'

interface TranscriptionMessage {
  type: string
  word?: string
  start_time?: number
  end_time?: number
  timestamp?: number
  text?: string
  paused?: boolean
  lang?: string
}

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [currentText, setCurrentText] = useState('')
  const [wsPort, setWsPort] = useState('8765')
  const [showSettings, setShowSettings] = useState(false)
  const [alwaysOnTop, setAlwaysOnTop] = useState(true)
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('')
  
  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const loadAudioDevices = async () => {
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true })
      
      const devices = await navigator.mediaDevices.enumerateDevices()
      const audioInputs = devices.filter(device => device.kind === 'audioinput')
      setAudioDevices(audioInputs)
      
      if (audioInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(audioInputs[0].deviceId)
      }
    } catch (error) {
      console.error('Failed to enumerate devices:', error)
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

  const startAudio = async () => {
    try {
      const constraints: MediaStreamConstraints = {
        audio: selectedDeviceId 
          ? { deviceId: { exact: selectedDeviceId } }
          : true
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      
      const source = audioContext.createMediaStreamSource(stream)
      sourceRef.current = source
      
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      
      processor.onaudioprocess = (e) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const inputSamples = e.inputBuffer.getChannelData(0)
          const inputSampleRate = audioContext.sampleRate
          const targetSampleRate = 24000
          
          const resampled = resample(inputSamples, inputSampleRate, targetSampleRate)
          wsRef.current.send(resampled.buffer)
        }
      }
      
      source.connect(processor)
      processor.connect(audioContext.destination)
      
      workletNodeRef.current = processor as any
      setIsListening(true)
    } catch (error) {
      console.error('Failed to start audio:', error)
      throw error
    }
  }

  const stopAudio = () => {
    setIsListening(false)
    
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
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

    try {
      const ws = new WebSocket(`ws://localhost:${wsPort}/`)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
      }

      ws.onclose = () => {
        setIsConnected(false)
        stopAudio()
      }

      ws.onerror = () => {
        setIsConnected(false)
      }

      ws.onmessage = (event) => {
        const message: TranscriptionMessage = JSON.parse(event.data)

        switch (message.type) {
          case 'word':
            setCurrentText(prev => prev + (message.word || '') + ' ')
            break
          case 'final':
            if (message.text) {
              setCurrentText(message.text)
              setTimeout(() => {
                setCurrentText('')
              }, 3000)
            }
            break
        }
      }
    } catch (error) {
      console.error('Failed to connect:', error)
      setIsConnected(false)
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

  const toggleListening = async () => {
    if (isListening) {
      stopAudio()
    } else {
      try {
        await startAudio()
      } catch (error) {
        console.error('Failed to start listening:', error)
      }
    }
  }

  const handleDragStart = async () => {
    try {
      await invoke('start_drag')
    } catch (error) {
      console.error('Failed to start drag:', error)
    }
  }

  const toggleAlwaysOnTop = async () => {
    try {
      const newValue = !alwaysOnTop
      await invoke('toggle_always_on_top', { onTop: newValue })
      setAlwaysOnTop(newValue)
    } catch (error) {
      console.error('Failed to toggle always on top:', error)
    }
  }

  useEffect(() => {
    loadAudioDevices()
    
    const timer = setTimeout(() => {
      connect()
    }, 500)

    return () => {
      clearTimeout(timer)
      disconnect()
    }
  }, [])

  return (
    <div className="h-screen w-full bg-black/80 backdrop-blur-md flex items-center relative">
      <div 
        className="flex-1 h-full flex items-center px-6 cursor-move"
        onMouseDown={handleDragStart}
      >
        <div className="w-full">
          {currentText ? (
            <p className="text-white text-2xl font-medium text-center drop-shadow-lg">
              {currentText}
            </p>
          ) : (
            <p className="text-white/50 text-lg text-center italic">
              {isConnected ? (isListening ? 'Listening...' : 'Click mic to start') : 'Connecting...'}
            </p>
          )}
        </div>
      </div>

      <div className="absolute top-2 right-2 flex gap-2">
        <Button
          size="sm"
          variant="ghost"
          className={`h-6 w-6 p-0 ${isListening ? 'text-green-500' : 'text-white/70'} hover:text-white hover:bg-white/10`}
          onClick={toggleListening}
          disabled={!isConnected}
        >
          {isListening ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="h-6 w-6 p-0 text-white/70 hover:text-white hover:bg-white/10"
          onClick={() => setShowSettings(true)}
        >
          <Settings className="h-4 w-4" />
        </Button>
      </div>

      <Dialog open={showSettings} onOpenChange={setShowSettings}>
        <DialogContent className="sm:max-w-[800px] bg-black/95 border-white/20 text-white">
          <DialogHeader>
            <DialogTitle>Settings</DialogTitle>
            <DialogDescription className="text-white/70">
              Configure your audio source and connection settings
            </DialogDescription>
          </DialogHeader>
          
          <div className="grid grid-cols-2 gap-6 py-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="audio-device">Audio Input Device</Label>
                <div className="flex gap-2">
                  <Select
                    value={selectedDeviceId}
                    onValueChange={setSelectedDeviceId}
                    disabled={isListening}
                  >
                    <SelectTrigger id="audio-device" className="bg-white/10 border-white/20">
                      <SelectValue placeholder="Select audio device" />
                    </SelectTrigger>
                    <SelectContent className="bg-black/95 border-white/20">
                      {audioDevices.map((device) => (
                        <SelectItem key={device.deviceId} value={device.deviceId}>
                          {device.label || `Device ${device.deviceId.substring(0, 8)}`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="icon"
                    variant="outline"
                    onClick={loadAudioDevices}
                    className="shrink-0 border-white/20 hover:bg-white/10"
                    disabled={isListening}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-xs text-white/50">
                  Select your microphone or system audio loopback device
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="ws-port">WebSocket Port</Label>
                <input
                  id="ws-port"
                  type="text"
                  value={wsPort}
                  onChange={(e) => setWsPort(e.target.value)}
                  className="flex h-10 w-full rounded-md border border-white/20 bg-white/10 px-3 py-2 text-sm text-white placeholder:text-white/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={isConnected}
                />
                <p className="text-xs text-white/50">
                  Port for eaRS WebSocket server (default: 8765)
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Window Behavior</Label>
                <div className="flex items-center justify-between rounded-md border border-white/20 bg-white/5 p-3">
                  <div>
                    <div className="text-sm font-medium">Always on Top</div>
                    <div className="text-xs text-white/50">Keep window above others</div>
                  </div>
                  <Button
                    size="sm"
                    variant={alwaysOnTop ? "default" : "outline"}
                    onClick={toggleAlwaysOnTop}
                    className={alwaysOnTop ? "" : "border-white/20"}
                  >
                    {alwaysOnTop ? 'On' : 'Off'}
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Connection Status</Label>
                <div className="space-y-2">
                  <div className="flex items-center justify-between rounded-md border border-white/20 bg-white/5 p-3">
                    <div className="flex items-center gap-2">
                      <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                      <div>
                        <div className="text-sm font-medium">WebSocket</div>
                        <div className="text-xs text-white/50">{isConnected ? 'Connected' : 'Disconnected'}</div>
                      </div>
                    </div>
                    {!isConnected && (
                      <Button size="sm" onClick={connect}>
                        Reconnect
                      </Button>
                    )}
                  </div>

                  <div className="flex items-center justify-between rounded-md border border-white/20 bg-white/5 p-3">
                    <div className="flex items-center gap-2">
                      <div className={`h-2 w-2 rounded-full ${isListening ? 'bg-green-500' : 'bg-gray-500'}`} />
                      <div>
                        <div className="text-sm font-medium">Audio Capture</div>
                        <div className="text-xs text-white/50">{isListening ? 'Active' : 'Stopped'}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default App
