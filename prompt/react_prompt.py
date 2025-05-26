from langchain.prompts import PromptTemplate


prompt = """
    This is prompt :> for react process 

"""




# thay bien context + question dc nhe :> 
def get_prompt(context: str, question: str): 


    return PromptTemplate(
        input_variables=["context", "question"],
        template=prompt
    ).format(
        context="{context}",
        question="{question}"
    )

