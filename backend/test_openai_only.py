# test_openai_only.py
import openai
import os
from dotenv import load_dotenv

def test_openai_connection():
    print("🧪 Testing OpenAI Connection...")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
            
        # Set API key (old format)
        openai.api_key = api_key
        
        print("✅ API key loaded from environment")
        
        # Test API call (old format)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"✅ API call successful: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_connection()