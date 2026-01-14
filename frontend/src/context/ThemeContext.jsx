import { createContext, useContext, useState, useEffect, useCallback } from 'react'

// Theme configurations based on dominant emotion
const EMOTION_THEMES = {
    // Stress/Anger → Calming Blue/Green
    anger: {
        name: 'calm',
        primary: 'from-blue-600 to-teal-500',
        background: 'bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900',
        accent: 'text-teal-400',
        border: 'border-teal-500/30',
        glow: 'shadow-teal-500/20',
        animationSpeed: 'slow',
        particles: 'calm-waves'
    },
    // Sadness → Warm Orange/Yellow
    sadness: {
        name: 'warm',
        primary: 'from-orange-500 to-yellow-400',
        background: 'bg-gradient-to-br from-slate-900 via-orange-950 to-slate-900',
        accent: 'text-orange-300',
        border: 'border-orange-400/30',
        glow: 'shadow-orange-400/20',
        animationSpeed: 'gentle',
        particles: 'floating-lights'
    },
    // Fear → Reassuring Purple/Lavender
    fear: {
        name: 'reassure',
        primary: 'from-purple-500 to-indigo-400',
        background: 'bg-gradient-to-br from-slate-900 via-purple-950 to-slate-900',
        accent: 'text-purple-300',
        border: 'border-purple-400/30',
        glow: 'shadow-purple-400/20',
        animationSpeed: 'gentle',
        particles: 'soft-glow'
    },
    // Happiness → Vibrant/Energetic
    happiness: {
        name: 'energetic',
        primary: 'from-yellow-400 to-pink-500',
        background: 'bg-gradient-to-br from-slate-900 via-yellow-950 to-slate-900',
        accent: 'text-yellow-300',
        border: 'border-yellow-400/30',
        glow: 'shadow-yellow-400/20',
        animationSpeed: 'lively',
        particles: 'sparkles'
    },
    // Neutral → Clean/Minimal
    neutral: {
        name: 'neutral',
        primary: 'from-gray-500 to-slate-400',
        background: 'bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900',
        accent: 'text-gray-300',
        border: 'border-gray-600/30',
        glow: 'shadow-gray-500/10',
        animationSpeed: 'normal',
        particles: 'subtle'
    },
    // Surprise → Electric/Dynamic
    surprise: {
        name: 'dynamic',
        primary: 'from-pink-500 to-cyan-400',
        background: 'bg-gradient-to-br from-slate-900 via-pink-950 to-slate-900',
        accent: 'text-pink-300',
        border: 'border-pink-400/30',
        glow: 'shadow-pink-400/20',
        animationSpeed: 'lively',
        particles: 'bursts'
    },
    // Disgust → Fresh/Clean Green
    disgust: {
        name: 'fresh',
        primary: 'from-green-500 to-emerald-400',
        background: 'bg-gradient-to-br from-slate-900 via-green-950 to-slate-900',
        accent: 'text-emerald-300',
        border: 'border-emerald-400/30',
        glow: 'shadow-emerald-400/20',
        animationSpeed: 'gentle',
        particles: 'leaves'
    }
}

// Default theme
const DEFAULT_THEME = EMOTION_THEMES.neutral

const ThemeContext = createContext({
    theme: DEFAULT_THEME,
    emotion: 'neutral',
    confidence: 0,
    engagement: 0.5,
    headPose: { yaw: 0, pitch: 0, roll: 0 },
    gaze: 'unknown',
    modalities: { audio: null, text: null, video: null },
    updateFromEmotion: () => {},
    isTransitioning: false
})

export function ThemeProvider({ children, socket }) {
    const [theme, setTheme] = useState(DEFAULT_THEME)
    const [emotion, setEmotion] = useState('neutral')
    const [confidence, setConfidence] = useState(0)
    const [engagement, setEngagement] = useState(0.5)
    const [headPose, setHeadPose] = useState({ yaw: 0, pitch: 0, roll: 0 })
    const [gaze, setGaze] = useState('unknown')
    const [modalities, setModalities] = useState({ audio: null, text: null, video: null })
    const [isTransitioning, setIsTransitioning] = useState(false)

    const updateFromEmotion = useCallback((emotionData) => {
        const newEmotion = emotionData.emotion?.toLowerCase() || 'neutral'
        
        // Only trigger transition if emotion actually changed
        if (newEmotion !== emotion) {
            setIsTransitioning(true)
            setTimeout(() => setIsTransitioning(false), 500)
        }

        setEmotion(newEmotion)
        setConfidence(emotionData.confidence || 0)
        setEngagement(emotionData.engagement || 0.5)
        setHeadPose(emotionData.head_pose || { yaw: 0, pitch: 0, roll: 0 })
        setGaze(emotionData.gaze || 'unknown')
        setModalities(emotionData.modalities || { audio: null, text: null, video: null })

        // Map emotion to theme
        const newTheme = EMOTION_THEMES[newEmotion] || DEFAULT_THEME
        setTheme(newTheme)
    }, [emotion])

    // Listen to WebSocket for emotion updates
    useEffect(() => {
        if (!socket) return

        const handleMessage = (event) => {
            try {
                const message = JSON.parse(event.data)
                if (message.type === 'emotion') {
                    updateFromEmotion(message)
                } else if (message.status === 'camera_stopped') {
                    // Reset to neutral when camera stops
                    updateFromEmotion({ emotion: 'neutral', confidence: 0 })
                }
            } catch (e) {
                // ignore parse errors
            }
        }

        socket.addEventListener('message', handleMessage)
        return () => socket.removeEventListener('message', handleMessage)
    }, [socket, updateFromEmotion])

    const value = {
        theme,
        emotion,
        confidence,
        engagement,
        headPose,
        gaze,
        modalities,
        updateFromEmotion,
        isTransitioning
    }

    return (
        <ThemeContext.Provider value={value}>
            {children}
        </ThemeContext.Provider>
    )
}

export function useTheme() {
    const context = useContext(ThemeContext)
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider')
    }
    return context
}

export { EMOTION_THEMES }
export default ThemeContext
