from langchain.prompts import PromptTemplate

FINAL_ANSWER_PROMPT = """
You are a reasoning expert responsible for producing the final answer to a complex query.

Input:
- [QUESTION]: the original complex question.
- [ANALYSIS]: the reasoning path that connects intermediate information to the final answer.
- [OBSERVATION]: a list of (sub-question, answer) pairs, showing available supporting information.

Instructions:
- Use the analysis and observations to determine the final answer.
- Return ONLY the exact information requested by the original question - nothing more.
- Your answer must be minimal and direct:
    * For a name: ONLY the name (e.g., "John Smith")
    * For a location: ONLY the place name (e.g., "Paris")
    * For a yes/no question: ONLY "Yes" or "No"
- Do NOT include:
    * Explanations or reasoning
    * Titles (Dr., Prof., etc.)
    * Additional context
    * Complete sentences
- If information is insufficient to answer, respond with:
    `"Insufficient information."`

IMPORTANT: To avoid JSON parsing errors, DO NOT use any double quotes (") or single quotes (') within your answer values. Use alternative wording if you need to mention something that would typically be in quotes.

Return your output using the following JSON structure:

final_answer = {{
        "answer": "Your answer here without using any quotes."
}}

---

### Few-shot Examples

Example 1 — Direct answer from single sub-question:

[QUESTION]:  
"Who developed the Mirage quantum chip?"

[ANALYSIS]:  
The development team is directly stated in one of the sub-answers.

[OBSERVATION]:  
- ("Who developed the Mirage quantum chip?", "Mirage was developed by NovaQ Systems.")

final_answer = {{
        "answer": "NovaQ Systems"
}}

---

Example 2 — Multi-hop reasoning:

[QUESTION]:  
"Which city is the headquarters of the company that created the Atlas-3 drone?"

[ANALYSIS]:  
We need to find the company that made Atlas-3, then trace the company's HQ location.

[OBSERVATION]:  
- ("Who created the Atlas-3 drone?", "It was built by AeroHelix.")
- ("Where is AeroHelix based?", "AeroHelix is headquartered in Seattle.")

final_answer = {{
        "answer": "Seattle"
}}

---

Example 3 — Parallel paths converging to one conclusion:

[QUESTION]:  
"Who is accused of fraud in both the TechDaily and FinJournal reports?"

[ANALYSIS]:  
We extract the individual mentioned in both sources regarding fraud accusations.

[OBSERVATION]:  
- ("Who does TechDaily accuse of fraud?", "Jordan Reeve was accused by TechDaily.")
- ("Who is mentioned in the FinJournal's fraud case?", "Jordan Reeve is central to the investigation.")

final_answer = {{
        "answer": "Jordan Reeve"
}}

---

Example 4 — Reverse dependency:

[QUESTION]:  
"Which university is affiliated with the researcher who developed the ReLume battery?"

[ANALYSIS]:  
First, identify the researcher, then find their university.

[OBSERVATION]:  
- ("Who developed the ReLume battery?", "It was developed by Elias Tan.")
- ("Where does Elias Tan work?", "He is a professor at Stanford University.")

final_answer = {{
        "answer": "Stanford University"
}}

---

Example 5 — No information available:

[QUESTION]:  
"Who designed the propulsion system of the Titan-X shuttle?"

[ANALYSIS]:  
None of the observations contain propulsion design information.

[OBSERVATION]:  
- ("Who funded the Titan-X shuttle?", "It was backed by OrbitalNet.")
- ("Where was the Titan-X shuttle assembled?", "At the CosmoTech facility in Nevada.")

final_answer = {{
        "answer": "Insufficient information."
}}

---

Now complete the following:

[QUESTION]:  
{question}

[ANALYSIS]:  
{analysis}

[OBSERVATION]:  
{observation}

final_answer =
"""


def get_final_answer_prompt(question: str, analysis: str, observation: list) -> str:

    formatted_obs = "\n".join([f"- ({q}, \"{a}\")" for q, a in observation])
    
    return PromptTemplate(
        input_variables=["question", "analysis", "observation"],
        template=FINAL_ANSWER_PROMPT
    ).format(
        question=question,
        analysis=analysis,
        observation=formatted_obs.strip()
    )
