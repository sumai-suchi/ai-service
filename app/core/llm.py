import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_fallback_llm():
    models = []
    
    # Groq (১ম পছন্দ - সর্বোচ্চ গতি)
    if os.getenv("GROQ_API_KEY"):
        models.append(
            ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
        )
        
    # OpenAI (২য় পছন্দ - স্থিতিশীল)
    if os.getenv("OPENAI_API_KEY"):
        models.append(
            ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
        )
        
    # Gemini (৩য় পছন্দ - ব্যাকআপ)
    if os.getenv("GEMINI_API_KEY"):
        models.append(
            ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=os.getenv("GEMINI_API_KEY"))
        )

    if not models:
        raise ValueError("কমপক্ষে একটি API Key কনফিগার করতে হবে!")
        
    primary = models[0]
    fallbacks = models[1:]
    
    return primary.with_fallbacks(fallbacks) if fallbacks else primary

llm = get_fallback_llm()