import { useState, useEffect, useRef } from 'react'

export default function TTSAudioPlayer({ socket }) {
    const audioRef = useRef(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [isMuted, setIsMuted] = useState(false)
    const [currentVoice, setCurrentVoice] = useState('')

    useEffect(() => {
        if (!socket) return

        const handleMessage = (event) => {
            try {
                const message = JSON.parse(event.data)
                if (message.type === 'tts_audio' && message.audio && !isMuted) {
                    playAudio(message.audio, message.format || 'mp3')
                    setCurrentVoice(message.voice || '')
                }
            } catch (e) {
                // ignore
            }
        }

        socket.addEventListener('message', handleMessage)
        return () => socket.removeEventListener('message', handleMessage)
    }, [socket, isMuted])

    const playAudio = (base64Audio, format) => {
        try {
            const audioBlob = base64ToBlob(base64Audio, `audio/${format}`)
            const audioUrl = URL.createObjectURL(audioBlob)
            
            if (audioRef.current) {
                audioRef.current.src = audioUrl
                audioRef.current.play()
                setIsPlaying(true)
            }
        } catch (e) {
            console.error('Audio playback error:', e)
        }
    }

    const base64ToBlob = (base64, contentType) => {
        const byteCharacters = atob(base64)
        const byteNumbers = new Array(byteCharacters.length)
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i)
        }
        const byteArray = new Uint8Array(byteNumbers)
        return new Blob([byteArray], { type: contentType })
    }

    const handleAudioEnd = () => {
        setIsPlaying(false)
    }

    const toggleMute = () => {
        setIsMuted(!isMuted)
        if (audioRef.current && !isMuted) {
            audioRef.current.pause()
            setIsPlaying(false)
        }
    }

    const stopAudio = () => {
        if (audioRef.current) {
            audioRef.current.pause()
            audioRef.current.currentTime = 0
            setIsPlaying(false)
        }
    }

    return (
        <div className="fixed bottom-4 right-4 z-50">
            <audio 
                ref={audioRef} 
                onEnded={handleAudioEnd}
                className="hidden"
            />
            
            <div className="flex items-center gap-2 bg-gray-800/90 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-700 shadow-lg">
                {/* Mute/Unmute button */}
                <button
                    onClick={toggleMute}
                    className={`p-2 rounded-full transition-colors ${
                        isMuted 
                            ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' 
                            : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                    }`}
                    title={isMuted ? 'Unmute TTS' : 'Mute TTS'}
                >
                    {isMuted ? '🔇' : '🔊'}
                </button>

                {/* Playing indicator */}
                {isPlaying && (
                    <>
                        <div className="flex items-center gap-1">
                            <span className="text-green-400 text-sm animate-pulse">
                                ▶️ Speaking
                            </span>
                            {/* Sound wave animation */}
                            <div className="flex items-center gap-0.5 h-4">
                                {[1, 2, 3, 4].map((i) => (
                                    <div
                                        key={i}
                                        className="w-1 bg-green-400 rounded-full animate-pulse"
                                        style={{
                                            height: `${Math.random() * 12 + 4}px`,
                                            animationDelay: `${i * 0.1}s`
                                        }}
                                    />
                                ))}
                            </div>
                        </div>
                        
                        {/* Stop button */}
                        <button
                            onClick={stopAudio}
                            className="p-1.5 bg-red-500/20 text-red-400 rounded-full hover:bg-red-500/30 transition-colors"
                            title="Stop"
                        >
                            ⏹️
                        </button>
                    </>
                )}

                {!isPlaying && !isMuted && (
                    <span className="text-gray-400 text-xs">TTS Ready</span>
                )}
            </div>
        </div>
    )
}
