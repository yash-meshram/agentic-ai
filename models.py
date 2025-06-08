from agno.models.groq import Groq
from agno.models.google import Gemini

def GroqModel():
    return Groq(id = "llama-3.3-70b-versatile")
    
def GeminiModel():
    return Gemini(id="gemini-2.0-flash")