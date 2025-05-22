import multiprocessing
from pathlib import Path
from openai import OpenAI
import os
import json
import re
from tqdm import tqdm

QUESTION_TYPE_CLASSIFICATION_PROMPT = """
### Instruction:

You are given a short story ("context"), followed by two questions: a hypothetical question about the story ("question 1") and a factual question based on details from the story ("question 2"). Your task is to identify the specific reasoning type required to answer each question based on the given taxonomy.

There are **9 possible reasoning types** you must choose from, each with its own definition and representative example. Carefully analyze the structure, intent, and required inference for each question before choosing the most appropriate reasoning type from the list.

### Reasoning Type Definitions:

1. **Condition** – The question introduces pre- or post-event conditions, often hypothetical or counterfactual.
     Example: *Jeff is a child with a very vivid sense of imagination. What is most likely to have happened next?*
2. **Causality** – The question asks about causes or effects of events.
     Example: *Which most likely caused the guests to avoid shards of glass?*
3. **Temporal** – The question involves reasoning about time or the sequence of events.
     Example: *Which is most likely if Chris later felt sick to his stomach?*
4. **Character** – The question focuses on the characters’ emotions, goals, or motivations.
     Example: *What outcome would be most upsetting to Ben?*
5. **Factoid** – The question requires retrieval of specific factual details.
     Example: *Where did people hide the money they got?*
6. **Abstraction** – The question involves drawing general conclusions, themes, or morals.
     Example: *What lesson did she learn from the passage?*
7. **Implication** – The question requires understanding implicit meanings, paraphrases, or indirect suggestions.
     Example: *Which answer implies Bob was pleased with his performance?*
8. **Perception** – The question involves the reader's judgments or values (e.g. moral or ethical).
     Example: *What is the most moral decision for Danielle?*
9. **Fictional** – The question constructs fully fictional or counterfactual scenarios.
     Example: *How does Dylan get home?*

### Output Format:

Identify the reasoning type of each question by choosing from the **exact reasoning type words** listed above (e.g., “Factoid”, “Causality”, etc.). Output strictly in the following format:
'''
question 1 reasoning type: <one of the 9 reasoning types>
question 2 reasoning type: <one of the 9 reasoning types>
'''

### Input:
context:
{context}
question1:
{question1}
question2:
{question2}
"""
client = OpenAI(
    api_key="xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key="xxx",
# )
# client = OpenAI(
#     api_key="xxx",
#     base_url="https://api.openai.com/v1",
# )


# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json")
input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_question_class.json")

params_list = [

    {"output_path":"/data/code/MoPS/assets/question_classification/qwen-max-2025-01-25_2-16.jsonl","model_name":"qwen-max-2025-01-25"},
    # {"output_path":"/data/code/MoPS/assets/question_classification/deepseek-chat-v3-0324.jsonl","model_name":"deepseek/deepseek-chat-v3-0324"},
    # {"output_path":"/data/code/MoPS/assets/question_classification/gpt-4o-2024-08-06_2-16.jsonl","model_name":"gpt-4o-2024-08-06"},


]
def get_response(
    client: OpenAI,
    content: str,
    model: str,
    temperature: float = 0.0,
):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    response = completion.choices[0].message.content
    assert isinstance(response, str)
    return response

def question_type_classification(input_path, output_path, client, model_name):
    with open(input_path, 'r') as f:
        data = json.load(f)

    results = []
    with open(output_path, "a") as f:
        for item in tqdm(data, desc="Processing items"):
            context = item['edited_context']
            question1 = item['question_1']
            question2 = item['question_2']

            prompt = QUESTION_TYPE_CLASSIFICATION_PROMPT.format(context=context, question1=question1, question2=question2)
            response = get_response(client, prompt, model_name)
            question1_reasoning_type_match = re.search(r"question 1 reasoning type:\s*(.*)", response, flags=re.IGNORECASE)
            question2_reasoning_type_match = re.search(r"question 2 reasoning type:\s*(.*)", response, flags=re.IGNORECASE)
            item["question1_reasoning_type"] = question1_reasoning_type_match.group(1).strip() if question1_reasoning_type_match else ""
            item["question2_reasoning_type"] = question2_reasoning_type_match.group(1).strip() if question2_reasoning_type_match else ""

            # print(item["question1_reasoning_type"])
            # print(item["question2_reasoning_type"])


            results.append(item)
            f.write(json.dumps(item) + "\n")


   #  with open(output_path, 'w') as f:
   #      json.dump(results, f)

# def run_action(params):
#     try:


#         question_type_classification(input_path, params["output_path"], client, params["model_name"])
#         return f"Success: {params['model_name']}"
#     except Exception as e:
#         return f"Error in {params['model_name']}: {str(e)}"


if __name__ == "__main__":
    # with multiprocessing.Pool(processes=len(params_list)) as pool:
    #     results = pool.map(run_action, params_list)

    # for result in results:
    #     print(result)
    question_type_classification(input_path, params_list[0]["output_path"], client, params_list[0]["model_name"])