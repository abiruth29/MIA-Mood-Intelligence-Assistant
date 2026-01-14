import { useState, useEffect } from 'react'

export default function TranscriptionDisplay({ socket }) {
    const [text, setText] = useState("")

    useEffect(() => {
        if (!socket) return

        const handleMessage = (event) => {
            try {
                const message = JSON.parse(event.data)
                if (message.type === 'transcription') {
                    // Append new text or replace? Let's just show the latest chunk for now, 
                    // or maybe a running history. For subtitles, latest chunk + fade is nice.
                    // Let's keep a running log of the last few sentences.
                    setText(prev => {
                        const newText = prev + " " + message.text
                        // Keep only last 200 chars to avoid overflow
                        return newText.slice(-200)
                    })
                } else if (message.status === 'camera_started') {
                    setText("") // Clear on start
                }
            } catch (e) {
                // ignore
            }
        }

        socket.addEventListener('message', handleMessage)

        return () => {
            socket.removeEventListener('message', handleMessage)
        }
    }, [socket])

    return (
        <div className="mt-6 p-4 bg-gray-800/80 backdrop-blur-sm rounded-xl border border-gray-700 max-w-2xl mx-auto min-h-[100px] flex flex-col justify-center">
            <h3 className="text-sm font-semibold text-gray-400 mb-2 uppercase tracking-wider">Live Transcription</h3>
            <p className="text-lg text-white font-medium leading-relaxed animate-pulse-slow">
                {text || <span className="text-gray-600 italic">Listening...</span>}
            </p>
        </div>
    )
}
