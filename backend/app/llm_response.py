"""
LLM Response Generator for MIA
Uses Ollama (local) or OpenAI API for generating empathetic responses.
"""

import asyncio
import httpx
from typing import Optional, Dict, Any
import json

class LLMResponseGenerator:
    """
    Generates contextual, empathetic responses based on:
    - Detected emotion
    - Transcribed text (user's speech)
    - Conversation history
    """
    
    # System prompts tailored to emotional states
    EMOTION_PROMPTS = {
        "anger": """You are MIA, a calm and understanding AI assistant. The user seems frustrated or angry.
Respond with patience, acknowledge their feelings, and help them feel heard.
Use a soothing tone. Don't be dismissive. Offer practical help if appropriate.""",
        
        "sadness": """You are MIA, a warm and compassionate AI assistant. The user seems sad or down.
Respond with empathy and gentle encouragement. Acknowledge their feelings without trying to immediately "fix" things.
Be supportive and let them know it's okay to feel this way.""",
        
        "fear": """You are MIA, a reassuring and supportive AI assistant. The user seems anxious or afraid.
Respond with calm reassurance. Help ground them. Offer perspective without dismissing their concerns.
Be steady and reliable in your tone.""",
        
        "happiness": """You are MIA, an enthusiastic and engaging AI assistant. The user seems happy!
Match their positive energy. Celebrate with them. Be warm and share in their joy.
Keep the conversation upbeat and encouraging.""",
        
        "surprise": """You are MIA, a curious and engaged AI assistant. The user seems surprised or intrigued.
Show interest in what surprised them. Be curious and engage with their discovery.
Match their sense of wonder.""",
        
        "disgust": """You are MIA, an understanding AI assistant. The user seems put off by something.
Acknowledge their reaction without judgment. Help them process if needed.
Be neutral and supportive.""",
        
        "neutral": """You are MIA, a friendly and helpful AI assistant.
Be conversational, helpful, and engaging. Respond naturally to whatever the user shares.
Be warm but not overly enthusiastic."""
    }
    
    def __init__(self, 
                 provider: str = "ollama",
                 model: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434",
                 openai_api_key: Optional[str] = None):
        """
        Initialize the LLM response generator.
        
        Args:
            provider: "ollama" or "openai"
            model: Model name (e.g., "llama3.2", "mistral", "gpt-3.5-turbo")
            ollama_url: URL for Ollama API
            openai_api_key: OpenAI API key (if using OpenAI)
        """
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url
        self.openai_api_key = openai_api_key
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges
        
        print(f"LLM Response Generator initialized: {provider}/{model}")
    
    def _get_system_prompt(self, emotion: str, confidence: float) -> str:
        """Get emotion-appropriate system prompt."""
        base_prompt = self.EMOTION_PROMPTS.get(emotion.lower(), self.EMOTION_PROMPTS["neutral"])
        
        # Add confidence context
        if confidence < 0.5:
            base_prompt += "\n\nNote: The emotion detection confidence is low, so be flexible in your response style."
        
        return base_prompt
    
    def _build_messages(self, 
                        user_text: str, 
                        emotion: str, 
                        confidence: float,
                        context: Optional[Dict] = None) -> list:
        """Build message list for the LLM."""
        messages = [
            {"role": "system", "content": self._get_system_prompt(emotion, confidence)}
        ]
        
        # Add conversation history
        for entry in self.conversation_history[-self.max_history:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        
        # Add current user message with context
        user_message = user_text
        if context:
            # Add subtle context about engagement/gaze if relevant
            if context.get("engagement", 0.5) < 0.3:
                user_message = f"[User seems distracted] {user_text}"
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def generate_response(self,
                                user_text: str,
                                emotion: str = "neutral",
                                confidence: float = 0.5,
                                context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate an empathetic response based on user input and emotional state.
        
        Args:
            user_text: What the user said (transcription)
            emotion: Detected emotion
            confidence: Emotion detection confidence
            context: Additional context (engagement, gaze, etc.)
            
        Returns:
            Dict with 'response', 'emotion_acknowledged', 'suggestions'
        """
        if not user_text or not user_text.strip():
            return {
                "response": "",
                "emotion_acknowledged": False,
                "suggestions": []
            }
        
        messages = self._build_messages(user_text, emotion, confidence, context)
        
        try:
            if self.provider == "ollama":
                response_text = await self._call_ollama(messages)
            elif self.provider == "openai":
                response_text = await self._call_openai(messages)
            else:
                response_text = "I'm here to help. What's on your mind?"
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_text,
                "assistant": response_text,
                "emotion": emotion,
                "confidence": confidence
            })
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            return {
                "response": response_text,
                "emotion_acknowledged": emotion != "neutral",
                "suggestions": self._generate_suggestions(emotion)
            }
            
        except Exception as e:
            print(f"LLM Error: {e}")
            # Fallback response based on emotion
            return {
                "response": self._get_fallback_response(emotion),
                "emotion_acknowledged": True,
                "suggestions": [],
                "error": str(e)
            }
    
    async def _call_ollama(self, messages: list) -> str:
        """Call Ollama API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 150  # Keep responses concise
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
    
    async def _call_openai(self, messages: list) -> str:
        """Call OpenAI API."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    def _get_fallback_response(self, emotion: str) -> str:
        """Get a fallback response when LLM is unavailable."""
        fallbacks = {
            "anger": "I hear you, and your frustration is valid. Take a moment if you need to.",
            "sadness": "I'm here with you. It's okay to feel this way.",
            "fear": "You're safe here. Let's take this one step at a time.",
            "happiness": "That's wonderful! I'm glad you're feeling good!",
            "surprise": "Oh! That's quite something, isn't it?",
            "disgust": "I understand that reaction. Would you like to talk about it?",
            "neutral": "I'm here and listening. What would you like to talk about?"
        }
        return fallbacks.get(emotion.lower(), fallbacks["neutral"])
    
    def _generate_suggestions(self, emotion: str) -> list:
        """Generate contextual suggestions based on emotion."""
        suggestions = {
            "anger": ["Would you like to try a quick breathing exercise?", "Want to talk about what's bothering you?"],
            "sadness": ["Would some calming music help?", "Would you like to share what's on your mind?"],
            "fear": ["Let's focus on the present moment.", "Would you like some grounding exercises?"],
            "happiness": ["What made your day great?", "Would you like to capture this moment?"],
            "surprise": ["Tell me more about what happened!", "How are you feeling about this?"],
            "disgust": ["Would you like to move on to something else?", "Is there anything I can help with?"],
            "neutral": ["How can I help you today?", "What's on your mind?"]
        }
        return suggestions.get(emotion.lower(), suggestions["neutral"])
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return {"turns": 0, "emotions": [], "summary": "No conversation yet"}
        
        emotions = [entry["emotion"] for entry in self.conversation_history]
        return {
            "turns": len(self.conversation_history),
            "emotions": emotions,
            "dominant_emotion": max(set(emotions), key=emotions.count),
            "last_exchange": self.conversation_history[-1] if self.conversation_history else None
        }


# Async streaming version for real-time responses
class StreamingLLMResponse:
    """Streaming version for word-by-word responses."""
    
    def __init__(self, generator: LLMResponseGenerator):
        self.generator = generator
    
    async def stream_response(self, 
                              user_text: str,
                              emotion: str = "neutral",
                              confidence: float = 0.5):
        """Stream response tokens for real-time display."""
        messages = self.generator._build_messages(user_text, emotion, confidence, None)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.generator.ollama_url}/api/chat",
                json={
                    "model": self.generator.model,
                    "messages": messages,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
