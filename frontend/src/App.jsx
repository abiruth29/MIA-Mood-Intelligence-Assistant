import { useState, useEffect } from 'react'
import MediaPreview from './components/MediaPreview'
import TranscriptionDisplay from './components/TranscriptionDisplay'
import { EmotionVisualizationWithSocket } from './components/EmotionVisualization'
import LLMResponseDisplay from './components/LLMResponseDisplay'
import TTSAudioPlayer from './components/TTSAudioPlayer'
import AnalyticsDashboard from './components/AnalyticsDashboard'
import { ThemeProvider, useTheme } from './context/ThemeContext'

// ─── SET THIS to Laptop A's local IP when running distributed ───────────────
// In development (single machine): leave as "localhost"
// In distributed mode: change to your laptop's IP, e.g. "192.168.1.100"
const BACKEND_HOST = import.meta.env.VITE_BACKEND_HOST || "localhost"
const BACKEND_PORT = import.meta.env.VITE_BACKEND_PORT || "8000"
const WS_URL = `ws://${BACKEND_HOST}:${BACKEND_PORT}/ws`
// ────────────────────────────────────────────────────────────────────────────

function AppContent({ socket, status, backendHost }) {
  const { theme, isTransitioning } = useTheme()
  const [showAnalytics, setShowAnalytics] = useState(false)

  return (
    <div className={`min-h-screen ${theme.background} text-white p-8 font-sans transition-all duration-700`}>
      <div className="max-w-4xl mx-auto text-center">
        {/* Header */}
        <div className="flex items-center justify-center gap-4 mb-2">
          <h1 className={`text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r ${theme.primary} tracking-tight transition-all duration-500`}>
            MIA
          </h1>
          <button
            onClick={() => setShowAnalytics(true)}
            className="p-2 bg-gray-800/50 hover:bg-gray-700/50 rounded-full transition-colors"
            title="View Analytics"
          >
            📊
          </button>
        </div>
        <p className={`text-xl mb-8 ${theme.accent} font-light transition-colors duration-500`}>
          Mood-Intelligence Assistant
        </p>

        <div className={`mb-2 inline-block px-4 py-1.5 rounded-full bg-gray-800/50 border ${theme.border} text-sm font-mono shadow-lg ${theme.glow} transition-all duration-500`}>
          Status: <span className={status === 'Connected' ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>{status}</span>
        </div>

        {/* Show which backend we're connected to */}
        <div className="mb-6 text-xs text-gray-600 font-mono">
          backend: {backendHost}
        </div>

        <div className="space-y-6">
          <MediaPreview socket={socket} />
          <EmotionVisualizationWithSocket socket={socket} />
          <LLMResponseDisplay socket={socket} />
          <TranscriptionDisplay socket={socket} />
        </div>
      </div>

      {/* TTS Audio Player (fixed position) */}
      <TTSAudioPlayer socket={socket} />

      {/* Analytics Dashboard Modal */}
      <AnalyticsDashboard
        isOpen={showAnalytics}
        onClose={() => setShowAnalytics(false)}
      />
    </div>
  )
}

function App() {
  const [status, setStatus] = useState('Connecting...')
  const [socket, setSocket] = useState(null)

  useEffect(() => {
    console.log(`Connecting to: ${WS_URL}`)
    const ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      setStatus('Connected')
      console.log('Connected to MIA Backend')
    }

    ws.onclose = () => {
      setStatus('Disconnected')
      console.log('Disconnected from MIA Backend')
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
      setStatus('Error')
    }

    setSocket(ws)

    return () => { ws.close() }
  }, [])

  return (
    <ThemeProvider socket={socket}>
      <AppContent socket={socket} status={status} backendHost={`${BACKEND_HOST}:${BACKEND_PORT}`} />
    </ThemeProvider>
  )
}

export default App
