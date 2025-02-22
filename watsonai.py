from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

class GenAI:
    """A class to interact with IBM Watson AI for text generation."""
    
    def __init__(self):
        """Initialize the GenAI instance with credentials and model parameters."""
     
        self.credentials = {
            "apikey": os.getenv("IBM_API_KEY"),
            "url": os.getenv("IBM_URL")
        }
        self.project_id = os.getenv("IBM_PROJECT_ID")
        
        if not all([self.credentials["apikey"], self.credentials["url"], self.project_id]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
   
        self.default_params = {
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.MIN_NEW_TOKENS: 5,
            GenParams.TEMPERATURE: 0.3,
            GenParams.TOP_K: 20,
            GenParams.TOP_P: 1
        }

        self.model = Model(
            model_id="ibm/granite-13b-instruct-v2",
            params=self.default_params,
            credentials=self.credentials,
            project_id=self.project_id
        )

        self.base_prompt = """You are a professional dietician named NourishWise. You provide evidence-based dietary advice, help people plan nutritious meals, and promote healthy eating habits. Your tone is friendly, supportive, and informative. You focus on providing practical and actionable advice.

Important Instructions:
- Always provide helpful responses based on well-established nutritional science
- Provide direct answers to user questions without asking follow-up questions
- If you're not completely certain about something, acknowledge what you do know and explain any limitations
- Never make up information or provide unsafe advice
- If a question is completely outside your expertise, politely explain that you can only provide advice about nutrition and dietary matters

User: {user_input}
Assistant: NourishWise: """

    def textgen(self, prompt: str, custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate text response based on the input prompt."""
        try:
            # Remove extra whitespace and ensure proper formatting
            personality_prompt = self.base_prompt.strip().format(user_input=prompt.strip())

            if custom_params:
                self.model.params.update(custom_params)

            response_text = ""
            generated_response = self.model.generate_text_stream(prompt=personality_prompt)
            
            if generated_response:
                for chunk in generated_response:
                    response_text += chunk
                    print(chunk, end='', flush=True) 
                return response_text
            else:
                return "No response generated"
                
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            print(error_msg)
            return error_msg

if __name__ == "__main__":
    try:
        ai = GenAI()
        while True:
            user_input = input("\nEnter your question (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using NourishWise! Goodbye!")
                break
                
            if not user_input.strip():
                print("Please enter a valid question.")
                continue
                
            print("\nGenerating response...\n")
            ai.textgen(user_input)  
            print("\n" + "-"*50)
            
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye!")