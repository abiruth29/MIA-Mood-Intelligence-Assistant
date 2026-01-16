import { useState, useEffect, useMemo } from 'react'

const API_BASE = 'http://localhost:8000/api'

// Simple bar chart component
function SimpleBarChart({ data, title, colorMap }) {
    const maxValue = Math.max(...Object.values(data), 1)
    
    return (
        <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-400 mb-3">{title}</h4>
            <div className="space-y-2">
                {Object.entries(data).map(([label, value]) => (
                    <div key={label} className="flex items-center gap-2">
                        <span className="w-20 text-xs text-gray-400 capitalize truncate">
                            {label}
                        </span>
                        <div className="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
                            <div
                                className={`h-full transition-all duration-500 ${colorMap?.[label] || 'bg-blue-500'}`}
                                style={{ width: `${(value / maxValue) * 100}%` }}
                            />
                        </div>
                        <span className="w-8 text-xs text-right font-mono text-gray-300">
                            {value}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    )
}

// Session history list
function SessionList({ sessions }) {
    if (!sessions || sessions.length === 0) {
        return (
            <div className="text-center text-gray-500 py-8">
                No sessions recorded yet
            </div>
        )
    }

    return (
        <div className="space-y-2 max-h-64 overflow-y-auto">
            {sessions.map((session, idx) => (
                <div 
                    key={session.session_id || idx}
                    className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                    <div>
                        <p className="text-sm font-medium text-white">
                            Session #{session.session_id?.slice(0, 8) || idx + 1}
                        </p>
                        <p className="text-xs text-gray-500">
                            {session.start_time ? new Date(session.start_time).toLocaleString() : 'Unknown time'}
                        </p>
                    </div>
                    <div className="text-right">
                        <p className="text-sm capitalize" style={{ color: getEmotionColor(session.dominant_emotion) }}>
                            {session.dominant_emotion || 'neutral'}
                        </p>
                        <p className="text-xs text-gray-500">
                            {session.total_turns || 0} turns
                        </p>
                    </div>
                </div>
            ))}
        </div>
    )
}

// Stat card component
function StatCard({ icon, label, value, subtext, color = 'blue' }) {
    const colorClasses = {
        blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30',
        green: 'from-green-500/20 to-green-600/10 border-green-500/30',
        purple: 'from-purple-500/20 to-purple-600/10 border-purple-500/30',
        orange: 'from-orange-500/20 to-orange-600/10 border-orange-500/30'
    }

    return (
        <div className={`bg-gradient-to-br ${colorClasses[color]} border rounded-xl p-4`}>
            <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl">{icon}</span>
                <span className="text-xs text-gray-400 uppercase tracking-wider">{label}</span>
            </div>
            <p className="text-3xl font-bold text-white">{value}</p>
            {subtext && <p className="text-xs text-gray-500 mt-1">{subtext}</p>}
        </div>
    )
}

function getEmotionColor(emotion) {
    const colors = {
        happiness: '#facc15',
        sadness: '#60a5fa',
        anger: '#ef4444',
        fear: '#a855f7',
        surprise: '#ec4899',
        disgust: '#22c55e',
        neutral: '#9ca3af'
    }
    return colors[emotion?.toLowerCase()] || colors.neutral
}

const EMOTION_COLORS = {
    happiness: 'bg-yellow-500',
    sadness: 'bg-blue-500',
    anger: 'bg-red-500',
    fear: 'bg-purple-500',
    surprise: 'bg-pink-500',
    disgust: 'bg-green-500',
    neutral: 'bg-gray-500'
}

export default function AnalyticsDashboard({ isOpen, onClose }) {
    const [emotionData, setEmotionData] = useState({})
    const [sessions, setSessions] = useState([])
    const [engagementStats, setEngagementStats] = useState({})
    const [dailySummary, setDailySummary] = useState({})
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState(null)
    const [selectedDays, setSelectedDays] = useState(7)

    const fetchAnalytics = async () => {
        setIsLoading(true)
        setError(null)
        
        try {
            const [emotionsRes, sessionsRes, engagementRes, dailyRes] = await Promise.all([
                fetch(`${API_BASE}/analytics/emotions?days=${selectedDays}`),
                fetch(`${API_BASE}/analytics/sessions?days=30`),
                fetch(`${API_BASE}/analytics/engagement?days=${selectedDays}`),
                fetch(`${API_BASE}/analytics/daily`)
            ])

            if (emotionsRes.ok) setEmotionData(await emotionsRes.json())
            if (sessionsRes.ok) setSessions(await sessionsRes.json())
            if (engagementRes.ok) setEngagementStats(await engagementRes.json())
            if (dailyRes.ok) setDailySummary(await dailyRes.json())
            
        } catch (err) {
            setError('Failed to fetch analytics data')
            console.error('Analytics fetch error:', err)
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        if (isOpen) {
            fetchAnalytics()
        }
    }, [isOpen, selectedDays])

    // Calculate totals
    const totalEmotions = useMemo(() => 
        Object.values(emotionData).reduce((a, b) => a + b, 0), 
        [emotionData]
    )
    
    const dominantEmotion = useMemo(() => {
        if (Object.keys(emotionData).length === 0) return 'neutral'
        return Object.entries(emotionData).sort((a, b) => b[1] - a[1])[0]?.[0] || 'neutral'
    }, [emotionData])

    if (!isOpen) return null

    return (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-gray-900 rounded-2xl border border-gray-700 w-full max-w-4xl max-h-[90vh] overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <div className="flex items-center gap-3">
                        <span className="text-2xl">📊</span>
                        <h2 className="text-xl font-bold text-white">Analytics Dashboard</h2>
                    </div>
                    <div className="flex items-center gap-4">
                        <select
                            value={selectedDays}
                            onChange={(e) => setSelectedDays(Number(e.target.value))}
                            className="bg-gray-800 text-white px-3 py-1.5 rounded-lg border border-gray-600 text-sm"
                        >
                            <option value={1}>Last 24 hours</option>
                            <option value={7}>Last 7 days</option>
                            <option value={30}>Last 30 days</option>
                        </select>
                        <button
                            onClick={fetchAnalytics}
                            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
                        >
                            🔄 Refresh
                        </button>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-gray-800 rounded-full transition-colors text-gray-400 hover:text-white"
                        >
                            ✕
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-12">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                        </div>
                    ) : error ? (
                        <div className="text-center py-12">
                            <p className="text-red-400 mb-4">{error}</p>
                            <button 
                                onClick={fetchAnalytics}
                                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg"
                            >
                                Try Again
                            </button>
                        </div>
                    ) : (
                        <>
                            {/* Stats Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                                <StatCard
                                    icon="🎭"
                                    label="Total Emotions"
                                    value={totalEmotions}
                                    subtext={`Last ${selectedDays} days`}
                                    color="purple"
                                />
                                <StatCard
                                    icon="💬"
                                    label="Sessions"
                                    value={sessions.length}
                                    subtext="Last 30 days"
                                    color="blue"
                                />
                                <StatCard
                                    icon="👀"
                                    label="Avg Engagement"
                                    value={`${Math.round((engagementStats.avg_engagement || 0) * 100)}%`}
                                    subtext={`Max: ${Math.round((engagementStats.max_engagement || 0) * 100)}%`}
                                    color="green"
                                />
                                <StatCard
                                    icon="😊"
                                    label="Dominant"
                                    value={dominantEmotion}
                                    subtext="Most frequent emotion"
                                    color="orange"
                                />
                            </div>

                            {/* Charts Row */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                                <SimpleBarChart
                                    data={emotionData}
                                    title="Emotion Distribution"
                                    colorMap={EMOTION_COLORS}
                                />
                                
                                <div className="bg-gray-800/50 rounded-lg p-4">
                                    <h4 className="text-sm font-semibold text-gray-400 mb-3">
                                        Today's Summary
                                    </h4>
                                    <div className="space-y-3">
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Date</span>
                                            <span className="text-white">{dailySummary.date || 'N/A'}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Sessions</span>
                                            <span className="text-white">{dailySummary.total_sessions || 0}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Conversations</span>
                                            <span className="text-white">{dailySummary.total_conversations || 0}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Avg Confidence</span>
                                            <span className="text-white">
                                                {Math.round((dailySummary.avg_confidence || 0) * 100)}%
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Session History */}
                            <div className="bg-gray-800/50 rounded-lg p-4">
                                <h4 className="text-sm font-semibold text-gray-400 mb-3">
                                    Recent Sessions
                                </h4>
                                <SessionList sessions={sessions} />
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
