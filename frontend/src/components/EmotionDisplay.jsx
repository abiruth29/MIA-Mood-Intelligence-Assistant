import { useState, useEffect } from 'react'

const EMOTION_CONFIG = {
    angry: { emoji: '😠', color: 'text-red-500', bg: 'bg-red-500/20' },
    disgust: { emoji: '🤢', color: 'text-green-600', bg: 'bg-green-600/20' },
    fear: { emoji: '😨', color: 'text-purple-500', bg: 'bg-purple-500/20' },
    happy: { emoji: '😊', color: 'text-yellow-400', bg: 'bg-yellow-400/20' },
    neutral: { emoji: '😐', color: 'text-gray-400', bg: 'bg-gray-400/20' },
    sad: { emoji: '😢', color: 'text-blue-400', bg: 'bg-blue-400/20' },
    surprise: { emoji: '😲', color: 'text-pink-400', bg: 'bg-pink-400/20' },
}

export default function EmotionDisplay({ socket }) {
    const [emotion, setEmotion] = useState('neutral')
    const [confidence, setConfidence] = useState(0)
    const [allScores, setAllScores] = useState([])

    useEffect(() => {
        if (!socket) return

        const handleMessage = (event) => {
            try {
                const message = JSON.parse(event.data)
                if (message.type === 'emotion') {
                    setEmotion(message.emotion.toLowerCase())
                    setConfidence(message.confidence)
                    setAllScores(message.all_scores || [])
                } else if (message.status === 'camera_stopped') {
                    setEmotion('neutral')
                    setConfidence(0)
                    setAllScores([])
                }
            } catch (e) {
                // ignore
            }
        }

        socket.addEventListener('message', handleMessage)
        return () => socket.removeEventListener('message', handleMessage)
    }, [socket])

    const config = EMOTION_CONFIG[emotion] || EMOTION_CONFIG.neutral

    return (
        <div className={`mt-6 p-6 rounded-xl border border-gray-700 max-w-2xl mx-auto ${config.bg} transition-all duration-500`}>
            <h3 className="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-wider">Voice Emotion</h3>

            <div className="flex items-center justify-center gap-6">
                <span className="text-6xl transition-all duration-300">{config.emoji}</span>
                <div className="text-left">
                    <p className={`text-3xl font-bold capitalize ${config.color} transition-colors duration-300`}>
                        {emotion}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                        Confidence: {(confidence * 100).toFixed(0)}%
                    </p>
                </div>
            </div>

            {allScores.length > 0 && (
                <div className="mt-4 grid grid-cols-4 gap-2 text-xs">
                    {allScores.slice(0, 4).map((s) => (
                        <div key={s.label} className="bg-gray-800/50 rounded px-2 py-1 text-center">
                            <span className="text-gray-400">{s.label}</span>
                            <span className="block text-white font-mono">{(s.score * 100).toFixed(0)}%</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
