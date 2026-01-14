import { useState, useEffect } from 'react'
import MediaPreview from './components/MediaPreview'
import TranscriptionDisplay from './components/TranscriptionDisplay'
import { EmotionVisualizationWithSocket } from './components/EmotionVisualization'
import { ThemeProvider, useTheme } from './context/ThemeContext'

function AppContent({ socket, status }) {
  const { theme, isTransitioning } = useTheme()

  return (
    <div className={`min-h-screen ${theme.background} text-white p-8 font-sans transition-all duration-700`}>
      <div className="max-w-4xl mx-auto text-center">
        <h1 className={`text-5xl font-extrabold mb-2 text-transparent bg-clip-text bg-gradient-to-r ${theme.primary} tracking-tight transition-all duration-500`}>
          MIA
        </h1>
        <p className={`text-xl mb-8 ${theme.accent} font-light transition-colors duration-500`}>
          Mood-Intelligence Assistant
        </p>

        <div className={`mb-6 inline-block px-4 py-1.5 rounded-full bg-gray-800/50 border ${theme.border} text-sm font-mono shadow-lg ${theme.glow} transition-all duration-500`}>
          Status: <span className={status === 'Connected' ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>{status}</span>
        </div>

        <div className="space-y-6">
          <MediaPreview socket={socket} />
          <EmotionVisualizationWithSocket socket={socket} />
          <TranscriptionDisplay socket={socket} />
        </div>
      </div>
    </div>
  )
}

function App() {
  const [status, setStatus] = useState('Connecting...')
  const [socket, setSocket] = useState(null)

  useEffect(() => {
    // Connect to the WebSocket server
    const ws = new WebSocket('ws://localhost:8000/ws')

    ws.onopen = () => {
      setStatus('Connected')
      console.log('Connected to Backend')
    }

    ws.onclose = () => {
      setStatus('Disconnected')
      console.log('Disconnected from Backend')
    }

    setSocket(ws)

    return () => {
      ws.close()
    }
  }, [])

  return (
    <ThemeProvider socket={socket}>
      <AppContent socket={socket} status={status} />
    </ThemeProvider>
  )
}

export default App
