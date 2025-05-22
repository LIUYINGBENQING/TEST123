import json
import re
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm 
from openai import AsyncOpenAI  
import os
from mops.utils import embedding, open_json, save_json, open_jsonl
import logging
import concurrent.futures 
from json_repair import repair_json

BACK_FORWARD_THINKING_PROMPT = """
You are a reasoning assistant.

Your task is to read a given paragraph composed of multiple sentences. For each sentence, generate:
1. **Forward reasoning questions** – about what may logically happen next as a result of this sentence.
2. **Backward reasoning questions** – about what might have caused or led to this sentence.

Then, organize all the questions across all sentences into a single, flat list.  
For each question, output the following in this exact format:

sentence: [The sentence the question is based on]  
type: [forward or backward]  
question: [The reasoning question]

---

### Input:
You will be provided with a paragraph in the variable `context`.

### Output format:
Return a list of reasoning questions, each formatted as shown above. Do NOT use JSON or any other structured data format.

---

### Example Input:
context:  
"Harold went for a long bike ride. He had a flat tire about five miles from home. Harold did not know how to fix a flat tire, which resulted in him having to walk all the way home with his bike. The long walk had its consequences."

---

### Expected Output:
sentence: Harold went for a long bike ride.  
type: forward  
question: What might Harold hope to achieve or experience on this bike ride?

sentence: Harold went for a long bike ride.  
type: backward  
question: What might have motivated Harold to go for a long bike ride?
[...]

The following is the context:  
{context}
"""


REASONING_SENTENCE_IDENTIFICATION_PROMPT = """
### Instruction:
You are provided with a story context consisting of multiple sentences. Your task is to analyze each sentence and identify which sentences require **forward reasoning** (predicting future consequences or actions) or **backward reasoning** (analyzing causes or background). 

To determine which sentences require reasoning:
1. **Forward reasoning**: Consider sentences describing decisions, events, or actions that have possible consequences or future developments.
2. **Backward reasoning**: Consider sentences that describe an event or action and need an explanation of why it happened or what caused it.

For each sentence in the context, decide whether it requires forward reasoning, backward reasoning, or neither.

### Output Format:
Do not include any '#' symbols or markdown headings in your output.
For each sentence in the context, output the following in order:

sentence: [The original sentence]  
reasoning_required: [Either "forward", "backward", or "none"]

### Example Input:
context:  
Harold went for a long bike ride. He had a flat tire about five miles from home. Harold did not know how to fix a flat tire, which resulted in him having to walk all the way home with his bike. The long walk had its consequences.

### Expected Output:
sentence: Harold went for a long bike ride.  
reasoning_required: none

sentence: He had a flat tire about five miles from home.  
reasoning_required: backward

sentence: Harold did not know how to fix a flat tire, which resulted in him having to walk all the way home with his bike.  
reasoning_required: forward

The following is the context:  
{context}
"""


SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT_WITH_QUESTION = """
### Instruction:
You are given a story context, a hypothetical question about the story, and four possible event options. Your task is to first deeply analyze the context and the hypothetical condition presented in the question by thinking through additional reasoning questions provided.

Specifically:
- Carefully consider each reasoning question listed below (both forward and backward reasoning questions).
- Do not include any '#' symbols or markdown headings in your output.
- Use these questions to understand potential causes, consequences, motivations, and logical developments in the story.
- After fully thinking through, infer which of the four options best aligns with both the hypothetical condition and the logical flow of the story.
**- Do not show your reasoning process. Only output the final answer and explanation.**

### Output Format (use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Input:
context:
{context}

hypothetical_question:
{question}

reasoning_questions:
{reasoning_questions}

options:
0: {option0},
1: {option1},
2: {option2},
3: {option3}
"""


INFERENCE_BETWEEN_OPTIONS_AND_SENTENCES_PROMPT = """
You are a narrative reasoning assistant. Your task is to analyze how each **option** is logically or narratively connected to specific **sentences in the story context**, *without using the question itself*.

The input contains:
- A **story context** of 3–6 sentences describing events.
- A **question** (to be ignored).
- Four **answer options**.

Your job:
For each option, examine which sentence(s) in the context it most directly builds upon or extends from, using narrative logic. Then, write a **brief potential story plot** that explains how this option could naturally follow from the relevant sentence(s) in the story context.

Use reasoning based on:
- **Causal relationships** (e.g., because X happened earlier, Y in the option happens).
- **Motivations or intentions** (e.g., a character acts due to something from the context).
- **References to people, items, or events** mentioned earlier.
- **Temporal/narrative continuation** (e.g., what happens next given what came before).

**Important constraints**:
- Ignore the question entirely.
- Do not compare options or judge correctness.
- Focus only on identifying a plausible continuation or reasoning based on the story context.

---

### Output Format:

Only output in the following format:

option 0 potential story plot :
option 1 potential story plot :
option 2 potential story plot :
option 3 potential story plot :

Each line should provide a short plot explanation that shows how the option logically follows from or connects to earlier sentence(s).

---

### Example Input:

context:  
"Harold went for a long bike ride. He had a flat tire about five miles from home. Harold did not know how to fix a flat tire, which resulted in him having to walk all the way home with his bike. The long walk had its consequences."  

question:  
"Which one would help Harold if he refused to learn how to fix the flat?"  

options:  
0: "As soon as he got home Harold looked for inflatable tires. The next day Harold woke up feeling refreshed and energized, ready to take on the day at work."  
1: "As soon as he got home Harold looked for inflatable tires. Then, Harold taught himself how to fix a flat."  
2: "As soon as he got home Harold looked for inflatable tires. The next day Harold woke up to blisters on his feet so he called out of work."  
3: "As soon as he got home Harold decided to repair the flat tire. The next day Harold woke up to blisters on his feet so he called out of work."

---

### Example Output:

option 0 potential story plot: Harold wanted to prevent future walking issues, so he searched for inflatable tires to be better prepared next time.  
option 1 potential story plot: After the exhausting walk home, Harold decided to learn how to fix a flat to avoid such situations in the future.  
option 2 potential story plot: The long walk home caused physical strain, and the next day Harold suffered blisters and couldn’t go to work.  
option 3 potential story plot: Despite deciding to fix the tire later, the walk had already taken a toll, leading to blisters the next morning.

---

### New Input:

context:  
{context}  
question:  
{question}  
options:  
0: {option0},
1: {option1},
2: {option2},
3: {option3},
"""



SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT_WITH_OPTIONS_RELATED_SENTENCES = """
### Instruction:
You are given a story context, a hypothetical question about the story, and four possible event options. Your task is to first deeply analyze the context and the hypothetical condition presented in the question by thinking through additional option potential story plots provided.

Specifically:
- Carefully consider each reasoning question listed below (both forward and backward reasoning questions).
- Use these questions to understand potential causes, consequences, motivations, and logical developments in the story.
- After fully thinking through, infer which of the four options best aligns with both the hypothetical condition and the logical flow of the story.

### Output Format (use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Input:
context:
{context}

hypothetical_question:
{question}

options:
0: {option0},
option 0 potential story plot: {option_0_potential_story_plot}
1: {option1},
option 1 potential story plot: {option_1_potential_story_plot}
2: {option2},
option 2 potential story plot: {option_2_potential_story_plot}
3: {option3},
option 3 potential story plot: {option_3_potential_story_plot}
"""

logger = logging.getLogger(__name__)

from mops.prompts import SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT


async def get_response(client, prompt, model_name, temperature=0.0, semaphore=None):
    if semaphore:
        async with semaphore:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
    else:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    response = completion.choices[0].message.content
    assert isinstance(response, str)
    return response


async def process_story(item, client, model_name, semaphore, progress_bar, data_list):
    option_order = item["shuffled_indices"]
    prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT.format(
        context=item["edited_context"],
        question=item["question_1"],
        option0=item["options"][0],
        option1=item["options"][1],
        option2=item["options"][2],
        option3=item["options"][3],
    )
    try:
        response = await get_response(client, prompt, model_name, semaphore=semaphore)
        selected_ending_match = re.search(r"answer:\s*(.*)", response)
        item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else "N/A"
        # if item["answer"] == "N/A":
        #     print("N/A response", response)
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else "No explanation"
        # print("item[answer]", item["answer"])
    except Exception as e:
        logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
        item["answer"] = "Error"
        item["explanation"] = "Error processing the request"
    
    progress_bar.update(1)  
    data_list.append(item)
    return item


async def inference_between_options_and_sentences(item, client, model_name, semaphore, progress_bar, data_list):
    option_order = item["shuffled_indices"]
    prompt = INFERENCE_BETWEEN_OPTIONS_AND_SENTENCES_PROMPT.format(
        context=item["edited_context"],
        question=item["question_1"],
        option0=item["options"][0],
        option1=item["options"][1],
        option2=item["options"][2],
        option3=item["options"][3],
    )
    try:
        response = await get_response(client, prompt, model_name, semaphore=semaphore)
        option_0_potential_story_plot_match = re.search(r"option 0 potential story plot:\s*(.*)", response)
        item["option_0_potential_story_plot"] = option_0_potential_story_plot_match.group(1).strip() if option_0_potential_story_plot_match else "N/A"
        option_1_potential_story_plot_match = re.search(r"option 1 potential story plot:\s*(.*)", response)
        item["option_1_potential_story_plot"] = option_1_potential_story_plot_match.group(1).strip() if option_1_potential_story_plot_match else "N/A"
        option_2_potential_story_plot_match = re.search(r"option 2 potential story plot:\s*(.*)", response)
        item["option_2_potential_story_plot"] = option_2_potential_story_plot_match.group(1).strip() if option_2_potential_story_plot_match else "N/A"
        option_3_potential_story_plot_match = re.search(r"option 3 potential story plot:\s*(.*)", response)
        item["option_3_potential_story_plot"] = option_3_potential_story_plot_match.group(1).strip() if option_3_potential_story_plot_match else "N/A"

        # print("item", item)
    except Exception as e:
        logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
        item["option_0_potential_story_plot"] = "Error"
        item["option_1_potential_story_plot"] = "Error"
        item["option_2_potential_story_plot"] = "Error"
        item["option_3_potential_story_plot"] = "Error"
    
    progress_bar.update(1)  
    data_list.append(item)
    return item

async def select_content_with_explicit_implicit_setting_with_options_related_sentences(item, client, model_name, semaphore, progress_bar, data_list):

    prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT_WITH_OPTIONS_RELATED_SENTENCES.format(
        context=item["edited_context"],
        question=item["question_1"],
        option0=item["options"][0],
        option1=item["options"][1],
        option2=item["options"][2],
        option3=item["options"][3],
        option_0_potential_story_plot=item["option_0_potential_story_plot"],
        option_1_potential_story_plot=item["option_1_potential_story_plot"],
        option_2_potential_story_plot=item["option_2_potential_story_plot"],
        option_3_potential_story_plot=item["option_3_potential_story_plot"],

    )
    try:
        response = await get_response(client, prompt, model_name, semaphore=semaphore)
        selected_ending_match = re.search(r"answer:\s*(.*)", response)
        item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else "N/A"
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else "No explanation"
    except Exception as e:
        logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
        item["answer"] = "Error"
        item["explanation"] = "Error processing the request"
    progress_bar.update(1)
    data_list.append(item)
    return item

async def add_single_sentence_question(item, client, model_name, semaphore, progress_bar, data_list):

    prompt = BACK_FORWARD_THINKING_PROMPT.format(context=item['edited_context'])

    try:
        response = await get_response(client, prompt, model_name, semaphore=semaphore)
        pattern = r"sentence:\s*(.*?)\s*type:\s*(.*?)\s*question:\s*(.*?)(?=\n\s*sentence:|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)

        new_questions = []
        for sentence, qtype, question in matches:
            new_questions.append({
                "sentence": sentence.strip(),
                "type": qtype.strip(),
                "question": question.strip()
            })
        item["new_questions"] = new_questions
        # print("new_questions", new_questions)
    except Exception as e:
        logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
        item["new_questions"] = ""
    progress_bar.update(1)
    data_list.append(item)
    return item

async def add_reasoning_sentence_identification(item, client, model_name, semaphore, progress_bar, data_list):
    prompt = REASONING_SENTENCE_IDENTIFICATION_PROMPT.format(context=item['edited_context'])

    try:
        response = await get_response(client, prompt, model_name, semaphore=semaphore)
        pattern = r"sentence:\s*(.*?)\s*reasoning_required:\s*(.*?)(?=\n\s*sentence:|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)

        reasoning_sentences = []
        for sentence, qtype in matches:
            reasoning_sentences.append({
                "sentence": sentence.strip(),
                "reasoning_required": qtype.strip()
            })
        item["reasoning_sentences"] = reasoning_sentences
    except Exception as e:
        logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
        item["reasoning_sentences"] = ""
    progress_bar.update(1)
    data_list.append(item)
    return item

async def select_content_with_new_questions(item, client, model_name, semaphore, progress_bar, data_list):
    question_list = []
    for q in item["new_questions"]:
        question_list.append(q["question"])
    prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT_WITH_QUESTION.format(
        context=item["edited_context"],
        question=item["question_1"],
        reasoning_questions=question_list,
        option0=item["options"][0],
        option1=item["options"][1],
        option2=item["options"][2],
        option3=item["options"][3],
    )
    try:
        response = await get_response(client, prompt, model_name, semaphore=semaphore)
        selected_ending_match = re.search(r"answer:\s*(.*)", response)
        item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else "N/A"
        # if item["answer"] == "N/A":
        #     print("N/A response", response)
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else "No explanation"
        # print("item[answer]", item["answer"])
    except Exception as e:
        logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
        item["answer"] = "Error"
        item["explanation"] = "Error processing the request"
    
    progress_bar.update(1)  
    data_list.append(item)
    return item

async def select_content_with_reasoning_questions(item, client, model_name, semaphore, progress_bar, data_list):

    question_list = []
    for question in item["new_questions"]:
        # 如果quetion["question"]在item["reasoning_sentences"]中reasoning_required等于forward或者backword则添加到forward_reasoning_sentences中
        for sentence in item["reasoning_sentences"]:
            if question["sentence"] == sentence["sentence"]:
                if sentence["reasoning_required"] != "none":
                    question_list.append(question["question"])
    prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT_WITH_QUESTION.format(
        context=item["edited_context"],
        question=item["question_1"],
        reasoning_questions=question_list,
        option0=item["options"][0],
        option1=item["options"][1],
        option2=item["options"][2],
        option3=item["options"][3],
    )
    try:
        response = await get_response(client, prompt, model_name, semaphore=semaphore)
        selected_ending_match = re.search(r"answer:\s*(.*)", response)
        item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else "N/A"
        # if item["answer"] == "N/A":
        #     print("N/A response", response)
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else "No explanation"
        # print("item[answer]", item["answer"])
    except Exception as e:
        logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
        item["answer"] = "Error"
        item["explanation"] = "Error processing the request"
    
    progress_bar.update(1)  
    data_list.append(item)
    return item

async def select_content_with_implicit_explicit_setting(input_path: Path, output_path: Path, temp_path: Path, client: AsyncOpenAI, model_name: str, semaphore: asyncio.Semaphore):
    stories = open_json(input_path) 
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Selecting content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        tasks = []

        with tqdm(total=len(stories), desc=f"Selecting content in {model_name}") as progress_bar:
            try:
                for item in stories:
                    tasks.append(process_story(item, client, model_name, semaphore, progress_bar, data_list))
                    # tasks.append(add_single_sentence_question(item, client, model_name, semaphore, progress_bar, data_list))
                    # tasks.append(inference_between_options_and_sentences(item, client, model_name, semaphore, progress_bar, data_list))
                    # tasks.append(select_content_with_explicit_implicit_setting_with_options_related_sentences(item, client, model_name, semaphore, progress_bar, data_list))
                    # tasks.append(add_reasoning_sentence_identification(item, client, model_name, semaphore, progress_bar, data_list))
                    # tasks.append(select_content_with_new_questions(item, client, model_name, semaphore, progress_bar, data_list))
                    # tasks.append(select_content_with_reasoning_questions(item, client, model_name, semaphore, progress_bar, data_list))


                results = await asyncio.gather(*tasks)

                for result in results:
                    f.write(json.dumps(result) + "\n")
            except (Exception, asyncio.CancelledError) as e:
                logger.error(f"Error processing item {item['mixed_question_id']}: {e}")
                progress_bar.close()
                print("data_list: ", data_list)
                with open(temp_path, "a") as f:
                    for item in data_list:
                        f.write(json.dumps(item) + "\n")    
                return


def run_action(params):
    try:
        input_path = Path(params["input_path"])
        output_path = Path(params["output_path"])
        model_name = params["model_name"]
        

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=params["api_key"],
        )

        # client = AsyncOpenAI(
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     api_key=params["api_key"],
        # )

        # client = AsyncOpenAI(
        #     api_key=params["api_key"],
        #     base_url="https://api.openai.com/v1",
        # )
        

        semaphore = asyncio.Semaphore(40)  
        

        asyncio.run(select_content_with_implicit_explicit_setting(input_path, output_path, params["temp_path"], client, model_name, semaphore))

        return f"Success: {model_name}"
    except Exception as e:
        return f"Error in {model_name}: {str(e)}"


def main():
    params_list = [
        {
        "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_gemini_pro.json",
        "output_path": "/data/code/MoPS/assets/explicit_implicit_setting_2246/gemini-2.5-pro-preview_new.jsonl",
        "temp_path": "/data/code/MoPS/assets/explicit_implicit_setting_2246/temp-gemini-2.5-pro-preview_new.jsonl",
        "model_name": "google/gemini-2.5-pro-preview",
        "api_key": "xxx"
        },        

        # {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/deepseek-chat-v3-0324-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/deepseek-chat-v3-0324-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/deepseek-chat-v3-0324-with-new-questions.json",
        # # "output_path": "/data/code/MoPS/assets/single_sentence_question/deepseek-chat-v3-0324-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/reasoning_sentences/deepseek-chat-v3-0324-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/deepseek-chat-v3-0324-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/deepseek-chat-v3-0324-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/deepseek-chat-v3-0324-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-deepseek-chat-v3-0324-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/temp-deepseek-chat-v3-0324-with-new-questions.jsonl",
        # "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-deepseek-chat-v3-0324-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/deepseek-chat-v3-0324-with-new-questions.jsonl",
        # "model_name": "deepseek/deepseek-chat-v3-0324",
        # "api_key": "xxx"
        # },


        # {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/llama3_1-8b-instruct-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/llama3_1-8b-instruct-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/llama3_1-8b-instruct-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-test.json",
        # #  "output_path": "/data/code/MoPS/assets/single_sentence_question/llama3_1-8b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/reasoning_sentences/llama3_1-8b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/llama3_1-8b-instruct-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/llama3_1-8b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/llama3_1-8b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/llama3_1-8b-instruct-with-new-questions.jsonl",

        #  "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-llama3_1-8b-instruct-with-new-questions.jsonl",
        #     # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-llama3_1-8b-instruct-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/temp-llama3_1-8b-instruct-with-new-questions.jsonl",
        #  "model_name": "meta-llama/llama-3.1-8b-instruct", 
        #  "api_key": "xxx"},

        # {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/llama3_1-70b-instruct-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/llama3_1-70b-instruct-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/llama3_1-70b-instruct-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-test.json",
        # #  "output_path": "/data/code/MoPS/assets/single_sentence_question/llama3_1-70b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/reasoning_sentences/llama3_1-70b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/llama3_1-70b-instruct-with-new-questions.jsonl",
        #  "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/llama3_1-70b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/llama3_1-70b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/llama3_1-70b-instruct-with-new-questions.jsonl",
        #  "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-llama3_1-70b-instruct-with-new-questions.jsonl",
        #     # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-llama3_1-70b-instruct-with-new-questions.jsonl",
        #  "model_name": "meta-llama/llama-3.1-70b-instruct", 
        #  "api_key": "xxx"},

        # {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/llama3_1-405b-instruct-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/llama3_1-405b-instruct-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/llama3_1-405b-instruct-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-test.json",
        # # "output_path": "/data/code/MoPS/assets/single_sentence_question/llama3_1-405b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/reasoning_sentences/llama3_1-405b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/llama3_1-405b-instruct-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/llama3_1-405b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/llama3_1-405b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/llama3_1-405b-instruct-with-new-questions.jsonl",
        #  "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-llama3_1-405b-instruct-with-new-questions.jsonl",
        #     # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-llama3_1-405b-instruct-with-new-questions.jsonl",
        #  "model_name": "meta-llama/llama-3.1-405b-instruct", 
        #  "api_key": "xxx"},

        # {
        #     # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        #     # "input_path": "/data/code/MoPS/assets/single_sentence_question/gemini-2.5-flash-preview-with-new-questions.json",
        #     "input_path": "/data/code/MoPS/assets/reasoning_sentences/gemini-2.5-flash-preview-with-new-questions.json",
        #     # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/gemini-2.5-flash-preview-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-test.json",
        # #  "output_path": "/data/code/MoPS/assets/single_sentence_question/gemini-2.5-flash-preview-with-new-questions.jsonl",
        # #   "output_path": "/data/code/MoPS/assets/reasoning_sentences/gemini-2.5-flash-preview-with-new-questions.jsonl",
        # #   "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/gemini-2.5-flash-preview-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/gemini-2.5-flash-preview-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/gemini-2.5-flash-preview-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/gemini-2.5-flash-preview-with-new-questions.jsonl",
        #  "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-gemini-2.5-flash-preview-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-gemini-2.5-flash-preview-with-new-questions.jsonl",
        #  "model_name": "google/gemini-2.5-flash-preview",
        #  "api_key": "xxx"},




        # {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/qwen-max-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/qwen-max-with-new-questions.json",
        # # "output_path": "/data/code/MoPS/assets/single_sentence_question/qwen-max-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/qwen-max-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/qwen-max-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/reasoning_sentences/qwen-max-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/qwen-max-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/qwen-max-with-new-questions.jsonl",
        #  "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-qwen-max-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-qwen-max-with-new-questions.jsonl",
        #  "model_name": "qwen-max-latest", 
        #  "api_key": "xxx"},

        # {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/qwen2.5-72b-instruct-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/qwen2.5-72b-instruct-with-new-questions.json",
        # #  "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/qwen2.5-72b-instruct-with-new-questions.json",
        # #  "output_path": "/data/code/MoPS/assets/single_sentence_question/qwen2.5-72b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/reasoning_sentences/qwen2.5-72b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/qwen2.5-72b-instruct-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/qwen2.5-72b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/qwen2.5-72b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/qwen2.5-72b-instruct-with-new-questions.jsonl",
        #  "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-qwen2.5-72b-instruct-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-qwen2.5-72b-instruct-with-new-questions.jsonl",
        #  "model_name": "qwen2.5-72b-instruct",
        #  "api_key": "xxx"},

        # {
        #     # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        #     # "input_path": "/data/code/MoPS/assets/single_sentence_question/qwen2.5-7b-instruct-with-new-questions.json",
        #     "input_path": "/data/code/MoPS/assets/reasoning_sentences/qwen2.5-7b-instruct-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/qwen2.5-7b-instruct-with-new-questions.json",
        # #  "output_path": "/data/code/MoPS/assets/single_sentence_question/qwen2.5-7b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/reasoning_sentences/qwen2.5-7b-instruct-with-new-questions.jsonl",
        # #  "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/qwen2.5-7b-instruct-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/qwen2.5-7b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/qwen2.5-7b-instruct-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/qwen2.5-7b-instruct-with-new-questions.jsonl",
        #  "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-qwen2.5-7b-instruct-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-qwen2.5-7b-instruct-with-new-questions.jsonl",
        #  "model_name": "qwen2.5-7b-instruct",
        #  "api_key": "xxx"},

        #  {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/gpt-4o-2024-08-06_2-16-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/gpt-4o-2024-08-06_2-16-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/gpt-4o-2024-08-06_2-16-with-new-questions.json",
        # # "output_path": "/data/code/MoPS/assets/single_sentence_question/gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/reasoning_sentences/gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-gpt-4o-2024-08-06_2-16-with-new-questions.jsonl",
        # "model_name": "gpt-4o-2024-08-06",
        # "api_key": "xxx"
        #  },

        # {
        # # "input_path": "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        # # "input_path": "/data/code/MoPS/assets/single_sentence_question/gpt-4o-mini-2024-07-18-with-new-questions.json",
        # "input_path": "/data/code/MoPS/assets/reasoning_sentences/gpt-4o-mini-2024-07-18-with-new-questions.json",
        # # "input_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/gpt-4o-mini-2024-07-18-with-new-questions.json",
        # # "output_path": "/data/code/MoPS/assets/single_sentence_question/gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/reasoning_sentences/gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_new_questions/gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # "output_path": "/data/code/MoPS/assets/select_content_with_reasoning_sentences/gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # # "output_path": "/data/code/MoPS/assets/select_content_with_options_related_sentences/gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # "temp_path": "/data/code/MoPS/assets/single_sentence_question/temp-gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # # "temp_path": "/data/code/MoPS/assets/inference_between_options_and_sentences/temp-gpt-4o-mini-2024-07-18-with-new-questions.jsonl",
        # "model_name": "gpt-4o-mini-2024-07-18",
        # "api_key": "xxx"
        #  },



    ]


    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_action, params_list))

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
