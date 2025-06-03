from langgraph.graph import StateGraph, END, START
from state import State, Action, ProcessingState
from prompt.analyze import get_analyze_prompt
from prompt.query_answer import get_query_answer_prompt
from prompt.query_decompose import get_query_decompose_prompt
from prompt.answer import get_final_answer_prompt
from helper import LLM, ReactOutputParse, QueryListOutputParser, AnswerOutputParser
from vectorstore import VectorStore
from time import sleep 

def initialize_node(state: State) -> State: 

    if state.question is None:
        raise ValueError("Question must be provided to initialize the state.")

    state.processing_state = ProcessingState.ANALYZE
    state.observation = []
    state.list_queries = []
    

    return state
 


def analyze_node(state: State) -> State: 

    prompt  = get_analyze_prompt(state.question, state.observation)
    llm = LLM.get_backbone_model(state.config.backbone)

    chain = llm | ReactOutputParse()

    try : 
        # sleep(5) 
        response = chain.invoke(prompt)
        state.react_output = response 
        

        return state

    except Exception as e:
        raise ValueError(f"Error during analysis: {e}")
        



def query_decompose_node(state: State) -> State:
    
    state.config.early_stopping -= 1
    prompt = get_query_decompose_prompt(state.question, state.react_output.analysis)
    llm = LLM.get_backbone_model(state.config.backbone)

    chain = llm | QueryListOutputParser()

    try : 
        # sleep(5) 
        response = chain.invoke(prompt)
        state.list_queries = response

        if not state.list_queries:
            raise ValueError("No queries generated from the decomposition step.")

        state.processing_state = ProcessingState.RAG

        return state
    except Exception as e:
        raise ValueError(f"Error during query decomposition: {e}")
    


def rag_node(state: State) -> State:

    vectorstore = VectorStore()

    for query in state.list_queries:
        try: 
            docs = vectorstore.similarity_search(query, k = state.config.k)
            formatted_docs = [f"Title: {doc.metadata.get('title', '')}\n Passage: {doc.page_content}" for doc in docs]
            
            prompt = get_query_answer_prompt(
                query = query, 
                information = formatted_docs,
            )
            llm = LLM.get_backbone_model(state.config.backbone)

            chain = llm | AnswerOutputParser()    
            # sleep(5) 
            response = chain.invoke(prompt)

            state.observation.append((query, response))
            state.processing_state = ProcessingState.ANALYZE
        
        except Exception as e:
            raise ValueError(f"Error during RAG process for query '{query}': {e}")

    return state

    
        
def generate_answer_node(state: State) -> State:

    prompt = get_final_answer_prompt(
        question=state.question, 
        analysis=state.react_output.analysis, 
        observation=state.observation
    ) 

    llm = LLM.get_backbone_model(state.config.backbone)

    chain = llm | AnswerOutputParser()

    try:
        # sleep(5) 
        response = chain.invoke(prompt)
        
        state.final_answer = response

    except Exception as e:
        raise ValueError(f"Error during final answer generation: {e}")

    return state


def router(state: State) -> str:

    if state.react_output.action == Action.RETRIEVE and state.config.early_stopping <= 0:
        state.processing_state = ProcessingState.GENERATE_ANSWER
        return "generate_answer"


    if state.react_output.action == Action.RETRIEVE:
        state.processing_state = ProcessingState.QUERY_DECOMPOSE
        return "query_decompose"
    
    elif state.react_output.action == Action.ANSWER:
        state.processing_state = ProcessingState.GENERATE_ANSWER
        return "generate_answer"


    raise ValueError(f"Invalid action: {state.react_output.action}. Expected 'query_decompose' or 'generate_answer'.")


def create_graph(): 

    workflow = StateGraph(State)

    workflow.add_node("initialize", initialize_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("query_decompose", query_decompose_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "analyze")
    workflow.add_conditional_edges(
        "analyze", 
        router, 
        {
            "query_decompose": "query_decompose",
            "generate_answer": "generate_answer"
        }

    )

    workflow.add_edge("query_decompose", "rag")
    workflow.add_edge("rag", "analyze")

    workflow.add_edge("generate_answer", END)

    return workflow.compile()



graph = create_graph()
