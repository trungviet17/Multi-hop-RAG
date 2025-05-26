from dotenv import load_dotenv
import os 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq 


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




class LLM : 

    @staticmethod 
    def get_gemini_model(model_name: str = "gemini-1.5-flash", max_token : int = 500, temperature: float = None) -> ChatGoogleGenerativeAI:

        return ChatGoogleGenerativeAI(
            model = model_name,
            max_tokens = max_token,
            temperature = temperature,
            api_key = GEMINI_API_KEY
        )



    @staticmethod 
    def get_gpt_model(model_name: str = "gpt-3.5-turbo", max_token : int = 500, temperature: float = None) -> ChatOpenAI:

        return ChatOpenAI(
            model = model_name,
            max_tokens = max_token,
            temperature = temperature,
            openai_api_key = GPT_API_KEY
        ) 
        


    @staticmethod
    def get_groq_model(model_name: str = "groq-1.5", max_token : int = 500, temperature: float = None) -> ChatGroq:
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