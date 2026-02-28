from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say: Working!"}],
    max_tokens=5
)

print(f"✓ Success! Response: {response.choices[0].message.content}")
print(f"Cost: ${response.usage.total_tokens * 0.000000375:.6f}")