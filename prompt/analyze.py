from langchain.prompts import PromptTemplate

ANALYZE_PROMPT = """
You are an expert in analyzing whether the current [OBSERVATION] provides sufficient information to answer the main [QUESTION].

The [QUESTION] is a multi-hop RAG query that requires complex reasoning, often involving intermediate inference steps.  
Each (question, answer) pair in [OBSERVATION] is a sub-question derived from the main question or from previous intermediate reasoning.  
These sub-questions are designed to help construct the answer step-by-step (via chains or parallel paths).

You must return:
1. "action": either "ANSWER" (if sufficient) or "RETRIEVE" (if more sub-questions are needed)
2. "analysis": your reasoning process. If more evidence is needed, include a strategic plan of what kind of sub-question or information would help complete the reasoning chain.

Respond in the following JSON format:
```json
{{
    "action": "ANSWER" or "RETRIEVE",
    "analysis": "..."
}}
```

# Few-shot Examples

## Example 1:
[QUESTION]:  
"Who currently controls QuantumMed after its acquisition by BioHelix?"  

[OBSERVATION]:  
- ("Who owns QuantumMed?", "QuantumMed is a subsidiary of BioHelix.")  
- ("Who acquired BioHelix in 2023?", "BioHelix was acquired by AstraZeneca in 2023.")  

Expected Output:
```json
{{
    "action": "ANSWER",
    "analysis": "QuantumMed is owned by BioHelix, which was acquired by AstraZeneca in 2023. Therefore, AstraZeneca currently controls QuantumMed."
}}
```

## Example 2:
[QUESTION]:  
"Which companies enabled the Nebula engine to support deep space missions?"  

[OBSERVATION]:  
- ("Who developed the Nebula engine's propulsion?", "IonCore developed the propulsion.")  
- ("Who designed its heat management system?", "MechaDesigns was responsible.")  
- ("What enables its suitability for deep space?", "The combination of IonCore's and MechaDesigns' technologies.")  

Expected Output:
```json
{{
    "action": "ANSWER",
    "analysis": "IonCore and MechaDesigns each contributed critical technologies. Their combined work makes the Nebula engine suitable for deep space. Thus, the two companies are the correct answer."
}}
```

## Example 3:
[QUESTION]:  
"Which Ivy League professor is leading the US government's AI policy task force?"  

[OBSERVATION]:  
- ("Who is leading the AI task force?", "Dr. Priya Natarajan.")  
- ("What is her current position?", "She is a professor of astrophysics.")  

Expected Output:
```json
{{
    "action": "RETRIEVE",
    "analysis": "We know Dr. Priya Natarajan is leading the task force and is a professor. But we don't yet know which university she is affiliated with, which is required to verify if she's from the Ivy League. We need to retrieve: 'Which university does she teach at?'"
}}
```

## Example 4:
[QUESTION]:
"Which country currently operates the space station module named Horizon-X?"

[OBSERVATION]:
- ("Who launched Horizon-X?", "The module was launched by AstroCorp.")
- ("Where is AstroCorp based?", "AstroCorp is a private space company.")

Expected Output:
```json
{{
  "action": "RETRIEVE",
  "analysis": "We know AstroCorp launched Horizon-X, and that it's a private company. However, we don't know which country it's registered in or affiliated with. To answer the main question, we need to retrieve: 'Which country is AstroCorp headquartered in or operates under?'"
}}
```

## Example 5:
[QUESTION]:
"Which tech CEO, who studied both AI and neuroscience, published a book in 2024 on consciousness?"

[OBSERVATION]:
- ("Who published a book on consciousness in 2024?", "Dr. Lila Kumar.")
- ("What is Dr. Lila Kumar's current role?", "CEO of NeuroSoft.")

Expected Output:
```json
{{
  "action": "RETRIEVE",
  "analysis": "We know Dr. Kumar is a CEO and published the book. However, we don't know her academic background. Since the question requires a CEO who studied both AI and neuroscience, we need to retrieve: 'What is Dr. Lila Kumar's educational background?'"
}}
```

Now evaluate the following input:

[QUESTION]: {question}
[OBSERVATION]: {observation}
"""





def get_analyze_prompt(question: str, observation: list):

    formatted_obs = ""
    for q, a in observation:
        formatted_obs += f"- ({q}, \"{a}\")\n"



    return PromptTemplate(
        input_variables=["question", "observation"],
        template=ANALYZE_PROMPT
    ).format(question=question, observation=formatted_obs.strip()) 


if __name__ == "__main__":
  
    question = "Who is the CEO of the company that developed the Atlas-3 drone?"
    observation = [
        ("Who created the Atlas-3 drone?", "It was built by AeroHelix."),
        ("Where is AeroHelix based?", "AeroHelix is headquartered in Seattle.")
    ]
    
    prompt = get_analyze_prompt(question, observation)
    print(prompt)
