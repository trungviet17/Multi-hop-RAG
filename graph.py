from langgraph.graph import StateGraph, END, START
from state import State


def initialize_node(state: State) -> State: 
    pass 


def analyze_node(state: State) -> State: 
    pass 



def query_decompose_node(state: State) -> State:
    pass



def rag_node(state: State) -> State:

    pass 



def generate_answer_node(state: State) -> State:
    pass



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
        {
            "query_decompose": lambda state: state.react_output.action == "query_decompose",
            "generate_answer": lambda state: state.react_output.action == "generate_answer"
        }
    )

    workflow.add_edge("query_decompose", "rag")
    workflow.add_edge("rag", "analyze")

    workflow.add_edge("generate_answer", END)

    return workflow.compile()



graph = create_graph()