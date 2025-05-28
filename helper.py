from dotenv import load_dotenv
import os 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq 
from langchain_core.output_parsers import BaseOutputParser
from state import ReactOutput, Action
import json 
import re 

load_dotenv('.env')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




class LLM : 

    @staticmethod 
    def get_gemini_model(model_name: str = "gemini-2.0-flash", max_token : int = 100, temperature: float = 0.3) -> ChatGoogleGenerativeAI:

        return ChatGoogleGenerativeAI(
            model = model_name,
            temperature = temperature,
            api_key = GEMINI_API_KEY
        )



    @staticmethod 
    def get_gpt_model(model_name: str = "gpt-3.5-turbo", max_token : int = 500, temperature: float = 0.3) -> ChatOpenAI:

        return ChatOpenAI(
            model = model_name,
            max_tokens = max_token,
            temperature = temperature,
            openai_api_key = GPT_API_KEY
        ) 
        


    @staticmethod
    def get_groq_model(model_name: str = "groq-1.5", max_token : int = 500, temperature: float = 0.3) -> ChatGroq:
        return ChatGroq(
            model = model_name,
            max_tokens = max_token,
            temperature = temperature,
            api_key = GROQ_API_KEY
        ) 
        pass


    @staticmethod
    def get_gemini_embedding_model(model_name: str = "text-embedding-004"):
        pass


    @staticmethod
    def get_gpt_embedding_model():
        pass



class ReactOutputParse(BaseOutputParser): 

    def parse(self, text: str) -> ReactOutput:
        try:

            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
            
            data = json.loads(text)

            action = data.get("action").strip().upper() if data.get("action") else None
            if action not in ["RETRIEVE", "ANSWER"]:
                raise ValueError(f"Invalid action: {action}. Expected one of 'RETRIEVE', 'ANSWER'.")

            return ReactOutput(
                action= Action.RETRIEVE if action == "RETRIEVE" else Action.ANSWER,
                analysis=data.get("analysis"),
            )
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output: {e}") from e
        


class QueryListOutputParser(BaseOutputParser): 

    def parse(self, text: str) -> list:
        try:
            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
            
            queries = json.loads(text)['queries']
            if not isinstance(queries, list):
                raise ValueError("Expected a list of queries.")
            
            return queries
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output: {e}") from e
        

class AnswerOutputParser(BaseOutputParser):

    def parse(self, text: str) -> str:
        try:
            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
            
            data = json.loads(text)
            answer = data.get("answer")
            if not answer:
                raise ValueError("Expected an 'answer' field in the output.")
            
            return answer
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output: {e}") from e
        

