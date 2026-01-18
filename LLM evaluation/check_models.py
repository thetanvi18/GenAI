import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API keys
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("=" * 60)
print("CHECKING AVAILABLE MODELS")
print("=" * 60)

# Check Groq models
print("\n🟠 GROQ MODELS:")
print("-" * 60)
try:
    models = groq_client.models.list()
    print(f"✅ Available Groq Models:")
    for model in models.data:
        print(f"   - {model.id}")
    
except Exception as e:
    print(f"❌ Error checking Groq models: {str(e)}")
    print("   Please check your GROQ_API_KEY in .env file")
    print("   Get a free key at: https://console.groq.com")

# Check Gemini models
print("\n🟢 GOOGLE GEMINI MODELS:")
print("-" * 60)
try:
    models = genai.list_models()
    
    # Filter for models that support generateContent
    generative_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
    
    print(f"✅ Available generative models: {len(generative_models)}")
    for model in generative_models:
        print(f"   - {model.name}")
    
except Exception as e:
    print(f"❌ Error checking Gemini models: {str(e)}")
    print("   Please check your GEMINI_API_KEY in .env file")

print("\n" + "=" * 60)
print("✨ Check complete!")
print("=" * 60)
