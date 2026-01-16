import { useState, useEffect } from 'react'

export default function LLMResponseDisplay({ socket }) {
    const [response, setResponse] = useState('')
    const [suggestions, setSuggestions] = useState([])
    const [isTyping, setIsTyping] = useState(false)
    const [displayedText, setDisplayedText] = useState('')

    useEffect(() => {
        if (!socket) return

        const handleMessage = (event) => {
            try {
                const message = JSON.parse(event.data)
                if (message.type === 'llm_response') {
                    setResponse(message.response)
                    setSuggestions(message.suggestions || [])
                    setIsTyping(true)
                    setDisplayedText('')
                }
            } catch (e) {
                // ignore
            }
        }

        socket.addEventListener('message', handleMessage)
        return () => socket.removeEventListener('message', handleMessage)
    }, [socket])

    // Typewriter effect
    useEffect(() => {
        if (!isTyping || !response) return

        let index = 0
        const timer = setInterval(() => {
            if (index < response.length) {
                setDisplayedText(response.slice(0, index + 1))
                index++
            } else {
                setIsTyping(false)
                clearInterval(timer)
            }
        }, 20) // 20ms per character

        return () => clearInterval(timer)
    }, [response, isTyping])

    if (!response && !displayedText) {
        return null // Don't show anything if no response yet
    }

    return (
        <div className="mt-6 p-5 bg-gradient-to-br from-indigo-900/50 to-purple-900/50 rounded-xl border border-indigo-500/30 max-w-2xl mx-auto">
            <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">🤖</span>
                <h3 className="text-sm font-semibold text-indigo-300 uppercase tracking-wider">
                    MIA Response
                </h3>
                {isTyping && (
                    <span className="ml-auto text-xs text-indigo-400 animate-pulse">
                        typing...
                    </span>
                )}
            </div>
            
            <p className="text-white text-lg leading-relaxed">
                {displayedText}
                {isTyping && <span className="animate-pulse">▋</span>}
            </p>

            {suggestions.length > 0 && !isTyping && (
                <div className="mt-4 pt-4 border-t border-indigo-500/20">
                    <p className="text-xs text-indigo-400 mb-2">Suggestions:</p>
                    <div className="flex flex-wrap gap-2">
                        {suggestions.map((suggestion, idx) => (
                            <button
                                key={idx}
                                className="px-3 py-1.5 text-sm bg-indigo-800/50 hover:bg-indigo-700/50 
                                         text-indigo-200 rounded-full transition-colors border border-indigo-600/30"
                            >
                                {suggestion}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
