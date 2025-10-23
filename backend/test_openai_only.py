# test_openai_only.py
import os

def test_openai_connection():
    print("🧪 Testing OpenAI Connection...")
    
    try:
        # First, try with older version approach
        import openai
        
        # Set API key
        openai.api_key = "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"
        
        print("✅ API key set with legacy method")
        
        # Test simple API call with legacy approach
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"✅ Legacy API call successful: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Legacy approach failed: {str(e)}")
        
        try:
            # Try new client approach
            os.environ["OPENAI_API_KEY"] = "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"
            
            client = openai.OpenAI()
            print("✅ New OpenAI client created successfully")
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10
            )
            
            print(f"✅ New API call successful: {response.choices[0].message.content}")
            return True
            
        except Exception as e2:
            print(f"❌ New approach also failed: {str(e2)}")
            return False

if __name__ == "__main__":
    test_openai_connection()