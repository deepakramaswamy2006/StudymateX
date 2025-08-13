import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return completion.choices[0].message.content