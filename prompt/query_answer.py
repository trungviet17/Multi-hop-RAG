from langchain.prompts import PromptTemplate



QUERY_ANSWER_PROMPT = """
You are an expert at answering complex sub-questions using retrieved information chunks.

Given:
- [QUERY]: the sub-question to answer.
- [INFORMATION]: a list of textual chunks (retrieved evidence), which may or may not contain enough data to answer the query.

Your task:
- Read the chunks carefully.
- Provide a concise and factual answer to the query.
- If the information is insufficient, respond with: "No information available."

IMPORTANT: If your response contains any double quotes (") within the "query" or "answer" values, you MUST escape them with a backslash (\\"). This ensures proper JSON formatting while preserving the original quotation marks.


Respond in the following JSON format:
```json
{{
  "query": "Your query here. Remember to escape any \\\"quotes\\\" inside this text",
  "answer": "Your answer here. Remember to escape any \\\"quotes\\\" inside this text"
}}
```
---

### Few-shot examples

Example 1:

[QUERY]:  
"Who founded Tesla?"

[INFORMATION]:  
chunk_1: Elon Musk is known for leading Tesla, but the company was originally founded by Martin Eberhard and Marc Tarpenning in 2003.  
chunk_2: Elon Musk later joined as an investor and became CEO.

Output:
```json
{{
  "query": "Who founded Tesla?",
  "answer": "Martin Eberhard and Marc Tarpenning"
}}
```
---

Example 2:

[QUERY]:  
"What is the battery capacity of the Tesla Model 3?"

[INFORMATION]:  
chunk_1: The Tesla Model 3 is a popular electric vehicle manufactured by Tesla.  
chunk_2: It comes in multiple configurations including Standard and Long Range.

Output:
```json
{{
  "query": "What is the battery capacity of the Tesla Model 3?",
  "answer": "No information available."
}}
```

---

Example 3:

[QUERY]:  
"When was OpenAI founded?"

[INFORMATION]:  
chunk_1: OpenAI was established with the goal of ensuring AGI benefits all of humanity.  
chunk_2: The company began its operations in December 2015 with backing from prominent tech entrepreneurs.

Output:
```json
{{
  "query": "When was OpenAI founded?",
  "answer": "December 2015"
}}
```
---

Now answer the following:

[QUERY]:  
{query}

[INFORMATION]:  
{information}

Output:
"""

def get_query_answer_prompt(query: str, information: list) -> str:
    
    information = "\n".join([f"chunk_{i+1}: {chunk}" for i, chunk in enumerate(information)])

    return PromptTemplate(
        input_variables=["query", "information"],
        template=QUERY_ANSWER_PROMPT
    ).format(query=query, information=information)


