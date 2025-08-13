import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

class IBMGraniteLLM:
    """
    IBM Granite 3.2-2B-Instruct model integration
    Supports both local loading and Hugging Face inference API
    """
    
    def __init__(self, use_local=True, device=None):
        """
        Initialize IBM Granite LLM
        
        Args:
            use_local (bool): Whether to load model locally or use HF API
            device (str): Device to load model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = "ibm-granite/granite-3.2-2b-instruct"
        self.use_local = use_local
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = None
        self.model = None
        
        if self.use_local:
            self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer locally"""
        try:
            print(f"Loading IBM Granite model: {self.model_name}")
            print(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print("IBM Granite model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading IBM Granite model: {e}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt according to IBM Granite's instruction format"""
        # IBM Granite uses a specific instruction format
        formatted_prompt = f"<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
        return formatted_prompt
    
    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0):
        """
        Generate response using IBM Granite
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated response
        """
        if self.use_local and self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
            
        try:
            if self.use_local:
                return self._generate_local(prompt, max_tokens, temperature)
            else:
                return self._generate_hf_api(prompt, max_tokens, temperature)
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float):
        """Generate using locally loaded model"""
        formatted_prompt = self._format_prompt(prompt)
        
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response.strip()
    
    def _generate_hf_api(self, prompt: str, max_tokens: int, temperature: float):
        """Generate using Hugging Face inference API"""
        from huggingface_hub import InferenceClient
        
        api_token = os.getenv("HF_API_TOKEN")
        if not api_token:
            raise ValueError("HF_API_TOKEN environment variable required for HF API usage")
        
        client = InferenceClient(
            model=self.model_name,
            token=api_token
        )
        
        formatted_prompt = self._format_prompt(prompt)
        
        response = client.text_generation(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False
        )
        
        return response.strip()
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_local": self.use_local,
            "model_loaded": self.model is not None if self.use_local else True
        }
