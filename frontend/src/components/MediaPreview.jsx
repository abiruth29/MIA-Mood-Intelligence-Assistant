import { useState, useEffect, useRef } from 'react'

export default function MediaPreview({ socket }) {
    const [imageSrc, setImageSrc] = useState(null)
    const [isStreaming, setIsStreaming] = useState(false)

    useEffect(() => {
        if (!socket) return

        const handleMessage = (event) => {
            try {
                const message = JSON.parse(event.data)

                if (message.type === 'video_frame') {
                    setImageSrc(`data:image/jpeg;base64,${message.data}`)
                } else if (message.status === 'camera_started') {
                    setIsStreaming(true)
                } else if (message.status === 'camera_stopped') {
                    setIsStreaming(false)
                    setImageSrc(null)
                }
            } catch (e) {
                console.error("Error parsing message:", e)
            }
        }

        socket.addEventListener('message', handleMessage)
        
        return () => {
            socket.removeEventListener('message', handleMessage)
        }
    }, [socket])

    const startCamera = () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send('start_camera')
        }
    }

    const stopCamera = () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send('stop_camera')
            // Force local state update since backend loop might block "stop" ack in this simple implementation
            setIsStreaming(false)
            setImageSrc(null)
        }
    }

    return (
        <div className="p-4 bg-gray-800 rounded-lg shadow-lg max-w-2xl mx-auto mt-8">
            <h2 className="text-2xl font-bold mb-4 text-blue-400">Live Camera Feed</h2>

            <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-4 flex items-center justify-center border border-gray-700">
                {imageSrc ? (
                    <img
                        src={imageSrc}
                        alt="Live Feed"
                        className="w-full h-full object-cover"
                    />
                ) : (
                    <div className="text-gray-500">Camera is off</div>
                )}
            </div>

            <div className="flex gap-4 justify-center">
                {!isStreaming ? (
                    <button
                        onClick={startCamera}
                        className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-full font-semibold transition-colors"
                    >
                        Start Camera
                    </button>
                ) : (
                    <button
                        onClick={stopCamera}
                        className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded-full font-semibold transition-colors"
                    >
                        Stop Camera
                    </button>
                )}
            </div>
        </div>
    )
}
