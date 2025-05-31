# Multi-hop-RAG

## Setup Env 

```
GEMINI_API_KEY = 

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=
LANGSMITH_PROJECT="Multi-hop RAG"


OPENAI_API_KEY = 
GROQ_API_KEY = 
QDRANT_API_KEY = 
QDRANT_URL = 
WANDB_API_KEY = 


```


## How-to-run 

```
python -m venv venv

pip install -r requirements.txt

python benchmark.py --id <your_id> --k <your_k> --backbone <your_backbone> --sample_size <your_sample_size> --early_stopping <your_early_stopping> 

```


## Run debug 

```
langgraph dev --port <your_port> 

```


