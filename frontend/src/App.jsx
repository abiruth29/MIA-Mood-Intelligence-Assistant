import { useState, useEffect } from 'react'

function App() {
  const [message, setMessage] = useState('Connecting...')
  const [socket, setSocket] = useState(null)

  useEffect(() => {
    // Connect to the WebSocket server
    const ws = new WebSocket('ws://localhost:8000/ws')

    ws.onopen = () => {
      setMessage('Connected to Backend')
      ws.send('Hello from Frontend!')
    }

    ws.onmessage = (event) => {
      console.log('Message from server:', event.data)
    }

    ws.onclose = () => {
      setMessage('Disconnected')
    }

    setSocket(ws)

    return () => {
      ws.close()
    }
  }, [])

  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4 text-blue-400">MIA</h1>
        <p className="text-xl mb-4">Mood-Intelligence Assistant</p>
        <div className="p-4 bg-gray-800 rounded-lg shadow-lg">
          <p className="font-mono text-green-400">Status: {message}</p>
        </div>
      </div>
    </div>
  )
}

export default App
