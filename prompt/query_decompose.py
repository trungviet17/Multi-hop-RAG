from langchain.prompts import PromptTemplate

QUERY_DECOMPOSE_PROMPT = """
You are an expert in multi-hop question answering query decomposition.

Given a main question and the previous step's analysis, your task is to generate a list of sub-questions or keywords that should be retrieved next to help answer the main question.

Output format (strictly) in the following JSON structure:
```json
{{
    "queries": [list of sub-questions or keywords]
}}
```
---

# Few-shot examples

## Example 1;

[QUESTION]:  
"Which company supplies the batteries used in Tesla Model S?"

[ANALYSIS]:  
"Current information mentions Tesla's models but does not specify the battery supplier. We need to find out which company supplies these batteries."

Output:
```json 
{{
    "queries": [
        "Who supplies batteries for Tesla Model S?",
        "Which companies manufacture batteries for electric vehicles?"
    ]
}}
```
## Example 2:

[QUESTION]:  
"Who is the author of the Pulitzer-winning book in 2010 that was also a former US ambassador to the UN?"

[ANALYSIS]:  
"The Pulitzer-winning book and the ambassador seem to be different entities. We need to find out the author of the Pulitzer-winning book, and whether that author was also an ambassador."
```json
Output:  
{{
    "queries": [
        "Who authored the Pulitzer-winning book in 2010?",
        "Who was the US ambassador to the UN in 2010?",
        "Did the Pulitzer-winning author serve as ambassador?"
    ]
}}
```
Given:

[QUESTION]: {question}  
[ANALYSIS]: {analysis}

Generate the output exactly as specified.
"""


def get_query_decompose_prompt(question: str, analysis: str) -> PromptTemplate:
    

    return PromptTemplate(
        input_variables=["question", "analysis"],
        template=QUERY_DECOMPOSE_PROMPT
    ).partial(question=question, analysis=analysis)
