import { useMemo, useState, useEffect } from 'react'
import { useTheme } from '../context/ThemeContext'

// Emotion configuration with emoji and colors
const EMOTION_CONFIG = {
    anger: { emoji: '😠', color: 'text-red-500', bg: 'bg-red-500/20', label: 'Anger' },
    disgust: { emoji: '🤢', color: 'text-green-600', bg: 'bg-green-600/20', label: 'Disgust' },
    fear: { emoji: '😨', color: 'text-purple-500', bg: 'bg-purple-500/20', label: 'Fear' },
    happiness: { emoji: '😊', color: 'text-yellow-400', bg: 'bg-yellow-400/20', label: 'Happy' },
    neutral: { emoji: '😐', color: 'text-gray-400', bg: 'bg-gray-400/20', label: 'Neutral' },
    sadness: { emoji: '😢', color: 'text-blue-400', bg: 'bg-blue-400/20', label: 'Sad' },
    surprise: { emoji: '😲', color: 'text-pink-400', bg: 'bg-pink-400/20', label: 'Surprise' },
}

// Bar chart component for emotion scores
function EmotionBarChart({ scores }) {
    const sortedScores = useMemo(() => {
        return [...scores].sort((a, b) => b.score - a.score)
    }, [scores])

    return (
        <div className="space-y-2">
            {sortedScores.map((item) => {
                const config = EMOTION_CONFIG[item.label] || EMOTION_CONFIG.neutral
                const percentage = Math.round(item.score * 100)
                
                return (
                    <div key={item.label} className="flex items-center gap-2">
                        <span className="w-20 text-xs text-gray-400 capitalize truncate">
                            {config.label || item.label}
                        </span>
                        <div className="flex-1 h-4 bg-gray-800 rounded-full overflow-hidden">
                            <div
                                className={`h-full ${config.bg} transition-all duration-500 ease-out`}
                                style={{ width: `${percentage}%` }}
                            >
                                <div className={`h-full ${config.color.replace('text-', 'bg-')} opacity-60`}
                                     style={{ width: `${Math.min(percentage * 2, 100)}%` }} />
                            </div>
                        </div>
                        <span className="w-10 text-xs text-right font-mono text-gray-300">
                            {percentage}%
                        </span>
                    </div>
                )
            })}
        </div>
    )
}

// Modality indicators (Audio/Text/Video)
function ModalityIndicator({ modalities }) {
    const indicators = [
        { key: 'audio', label: '🎤 Audio', value: modalities?.audio },
        { key: 'text', label: '📝 Text', value: modalities?.text },
        { key: 'video', label: '📹 Video', value: modalities?.video },
    ]

    return (
        <div className="flex justify-center gap-3 text-xs">
            {indicators.map(({ key, label, value }) => (
                <div
                    key={key}
                    className={`px-2 py-1 rounded-full ${
                        value ? 'bg-gray-700 text-white' : 'bg-gray-800/50 text-gray-600'
                    }`}
                >
                    {label}: <span className="capitalize font-medium">{value || '—'}</span>
                </div>
            ))}
        </div>
    )
}

// Engagement meter
function EngagementMeter({ engagement, gaze }) {
    const percentage = Math.round(engagement * 100)
    const gazeEmoji = {
        'engaged': '👀',
        'looking_away': '👁️‍🗨️',
        'looking_up': '🔼',
        'looking_down': '🔽',
        'partially_engaged': '👁️',
        'unknown': '❓'
    }

    return (
        <div className="flex items-center gap-3 text-xs">
            <span className="text-gray-500">Engagement:</span>
            <div className="w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                    className={`h-full transition-all duration-300 ${
                        engagement > 0.7 ? 'bg-green-500' :
                        engagement > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
            <span className="font-mono text-gray-400">{percentage}%</span>
            <span title={`Gaze: ${gaze}`}>{gazeEmoji[gaze] || gazeEmoji.unknown}</span>
        </div>
    )
}

// Head pose visualization (simple pitch/yaw indicator)
function HeadPoseIndicator({ headPose }) {
    const { yaw, pitch } = headPose || { yaw: 0, pitch: 0 }
    
    // Clamp values for display
    const clampedYaw = Math.max(-45, Math.min(45, yaw))
    const clampedPitch = Math.max(-45, Math.min(45, pitch))
    
    return (
        <div className="flex items-center gap-2 text-xs text-gray-500">
            <span>Head:</span>
            <div className="relative w-8 h-8 bg-gray-800 rounded-full border border-gray-700">
                {/* Center dot that moves based on head pose */}
                <div
                    className="absolute w-2 h-2 bg-blue-400 rounded-full transition-all duration-200"
                    style={{
                        left: `calc(50% + ${clampedYaw * 0.3}px - 4px)`,
                        top: `calc(50% + ${clampedPitch * 0.3}px - 4px)`
                    }}
                />
            </div>
            <span className="font-mono text-gray-600">
                Y:{Math.round(yaw)}° P:{Math.round(pitch)}°
            </span>
        </div>
    )
}

export default function EmotionVisualization() {
    const { emotion, confidence, engagement, gaze, headPose, modalities, theme, isTransitioning } = useTheme()
    
    const config = EMOTION_CONFIG[emotion] || EMOTION_CONFIG.neutral
    
    // Get all_scores from theme context or use defaults
    const [allScores, setAllScores] = useState([])
    
    // We need to also listen to the raw scores - let's get them from the socket
    // For now, generate placeholder scores based on confidence
    const displayScores = useMemo(() => {
        if (allScores.length > 0) return allScores
        
        // Generate placeholder based on current emotion
        const emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        return emotions.map(emo => ({
            label: emo,
            score: emo === emotion ? confidence : (1 - confidence) / 6
        }))
    }, [allScores, emotion, confidence])

    return (
        <div className={`mt-6 p-6 rounded-xl border ${theme.border} max-w-2xl mx-auto ${config.bg} 
                        transition-all duration-500 ${isTransitioning ? 'scale-[1.02]' : 'scale-100'}`}>
            
            {/* Header */}
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                    Tri-Modal Emotion Analysis
                </h3>
                <HeadPoseIndicator headPose={headPose} />
            </div>

            {/* Main emotion display */}
            <div className="flex items-center justify-center gap-6 mb-6">
                <span className={`text-7xl transition-all duration-300 ${isTransitioning ? 'scale-125' : 'scale-100'}`}>
                    {config.emoji}
                </span>
                <div className="text-left">
                    <p className={`text-4xl font-bold capitalize ${config.color} transition-colors duration-300`}>
                        {config.label || emotion}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                        Confidence: <span className="font-mono">{(confidence * 100).toFixed(0)}%</span>
                    </p>
                </div>
            </div>

            {/* Modality indicators */}
            <div className="mb-4">
                <ModalityIndicator modalities={modalities} />
            </div>

            {/* Engagement meter */}
            <div className="flex justify-center mb-4">
                <EngagementMeter engagement={engagement} gaze={gaze} />
            </div>

            {/* Emotion bar chart */}
            <div className="mt-4 p-4 bg-gray-900/50 rounded-lg">
                <h4 className="text-xs text-gray-500 mb-3 uppercase tracking-wide">Emotion Breakdown</h4>
                <EmotionBarChart scores={displayScores} />
            </div>
        </div>
    )
}

// Also export a hook-aware version that gets scores from socket
export function EmotionVisualizationWithSocket({ socket }) {
    const { emotion, confidence, engagement, gaze, headPose, modalities, theme, isTransitioning } = useTheme()
    const [allScores, setAllScores] = useState([])

    useEffect(() => {
        if (!socket) return

        const handleMessage = (event) => {
            try {
                const message = JSON.parse(event.data)
                if (message.type === 'emotion' && message.all_scores) {
                    setAllScores(message.all_scores)
                }
            } catch (e) {
                // ignore
            }
        }

        socket.addEventListener('message', handleMessage)
        return () => socket.removeEventListener('message', handleMessage)
    }, [socket])

    const config = EMOTION_CONFIG[emotion] || EMOTION_CONFIG.neutral
    
    const displayScores = useMemo(() => {
        if (allScores.length > 0) return allScores
        const emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        return emotions.map(emo => ({
            label: emo,
            score: emo === emotion ? confidence : (1 - confidence) / 6
        }))
    }, [allScores, emotion, confidence])

    return (
        <div className={`mt-6 p-6 rounded-xl border ${theme.border} max-w-2xl mx-auto ${config.bg} 
                        transition-all duration-500 ${isTransitioning ? 'scale-[1.02]' : 'scale-100'}`}>
            
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                    Tri-Modal Emotion Analysis
                </h3>
                <HeadPoseIndicator headPose={headPose} />
            </div>

            <div className="flex items-center justify-center gap-6 mb-6">
                <span className={`text-7xl transition-all duration-300 ${isTransitioning ? 'scale-125' : 'scale-100'}`}>
                    {config.emoji}
                </span>
                <div className="text-left">
                    <p className={`text-4xl font-bold capitalize ${config.color} transition-colors duration-300`}>
                        {config.label || emotion}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                        Confidence: <span className="font-mono">{(confidence * 100).toFixed(0)}%</span>
                    </p>
                </div>
            </div>

            <div className="mb-4">
                <ModalityIndicator modalities={modalities} />
            </div>

            <div className="flex justify-center mb-4">
                <EngagementMeter engagement={engagement} gaze={gaze} />
            </div>

            <div className="mt-4 p-4 bg-gray-900/50 rounded-lg">
                <h4 className="text-xs text-gray-500 mb-3 uppercase tracking-wide">Emotion Breakdown</h4>
                <EmotionBarChart scores={displayScores} />
            </div>
        </div>
    )
}
