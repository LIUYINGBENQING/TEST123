import random
import re
import uuid
from pathlib import Path
from typing import List
import argparse
import os
import json
import time

import numpy as np
import tyro
from ndicts import NestedDict
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from tqdm import tqdm
from http import HTTPStatus

from mops.cal_score_distribution import calculate_bleu_score, compute_bertscore
from mops.constants import client, logger, openai_model, aliyun_model
from mops.prompts import (
    BACKGROUND_PROMPT,
    ENDING_PROMPT,
    EVENT_PROMPT,
    PROTAGONIST_ANTAGONIST_PROMPT,
    PROTAGONIST_DEUTERAGONIST_PROMPT,
    PROTAGONIST_PROMPT,
    TWIST_PROMPT,
    EDIT_POSSIBLE_STORY_PROMPT,
    SELECT_ENDING_PROMPT,
    TEST_POSSIBLE_STORY_PROMPT,
    SELECT_REASONING_TYPE_PROMPT,
    FORWARD_CHAINING_EDIT_STORY_PROMPT,
    BACKWARD_CHAINING_EDIT_STORY_PROMPT,
    SELECT_CONTENT_PROMPT,
    GENERATE_STORY_CONTENT_PROMPT,
    SELECT_CONTENT_PROMPT_HARD,
    GENERATE_INVERSE_OPTION_PROMPT,
    DETECT_OPTION_CONTRACTION_PROMPT,
    DETECT_QUESTION_CONTRACTION_PROMPT,
    DETECT_OPTION_ISSUE_PROMPT,
    EDIT_CONTEXT_PROMPT,
    OPTION_OPTIMIZATION_VERIFICATION_PROMPT,
    SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT,
    SELECT_CONTENT_WITH_IMPLICIT_SETTING_PROMPT,
    SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_COT_PROMPT,
    SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_ONE_SHOT_PROMPT,
    SELECT_CONTENT_WITH_EXPLICIT_SETTING_PROMPT,
    SELECT_CONTENT_WITH_DOUBLE_EXPLICIT_SETTING_PROMPT,

)
from mops.utils import embedding, open_json, save_json, open_jsonl
import dashscope


def get_response(
    client: OpenAI,
    content: str,
    # model: str = openai_model,
    model: str = aliyun_model,
    # model: str = args.model_name,
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

def get_response_dashscope(
    client: OpenAI,
    content: str,
    model: str = aliyun_model,
    # model: str = args.model_name,
    temperature: float = 0.0,
):
    messages=[{"role": "user", "content": content}]
    response = dashscope.Generation.call(
        api_key=os.getenv('ALIYUN_API_KEY'),
        model=model,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        temperature=temperature,
    )

    assert isinstance(response, str)
    return response
def filter_serial_numbers(texts: List[str]):
    # filter serial number 1. 2. 3. ...
    pattern = r"\d+\."
    result = []
    for text in texts:
        replaced_text = re.sub(pattern, "", text)
        result.append(replaced_text.strip())
    return result


# def pair_deduplicate(texts_a: List[str], texts_b: List[str], threshold: float = 0.85):
#     """Remove duplicate texts from `texts_a` based on
#     their embedding similarity to texts in `texts_b`."""

#     if len(texts_b) == 0:
#         return texts_a

#     # Remove fixed formats from the string
#     # for accurately calculating embedding similarity
#     pure_texts_a = [
#         text.lower()
#         .replace("deuteragonist:", "")
#         .replace("protagonist:", "")
#         .replace("antagonist:", "")
#         .replace("the ending of the narrative is to explore", "")
#         for text in texts_a
#     ]
#     pure_texts_b = [
#         text.lower()
#         .replace("deuteragonist:", "")
#         .replace("protagonist:", "")
#         .replace("antagonist:", "")
#         .replace("the ending of the narrative is to explore", "")
#         for text in texts_b
#     ]
#     embedding_array_a = embedding(pure_texts_a)
#     embedding_array_b = embedding(pure_texts_b)
#     sim_matrix = cos_sim(embedding_array_a, embedding_array_b)
#     to_keep_a = np.ones(len(texts_a), dtype=bool)
#     for i in range(sim_matrix.shape[0]):
#         if sim_matrix[i].max() > threshold:
#             to_keep_a[i] = False
#     keeped_texts_a = []
#     for idx, is_keep in enumerate(to_keep_a):
#         if is_keep:
#             keeped_texts_a.append(texts_a[idx])
#     return keeped_texts_a





def edit_possible_story(
    story_path: Path,
    edit_story_path: Path,
    client: OpenAI,
):
    stories = open_jsonl(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Edit possible story in: {edit_story_path}")

    edited_story_list = []
    for item in tqdm(stories, desc="Editing possible story"):
        gold_label = item["gold_label"]
        prompt = EDIT_POSSIBLE_STORY_PROMPT.format(
            story=item["document"],
            question=item["question"],
            ending=item["options"][gold_label],
        )
        
        response = get_response(client, prompt)
        edited_story_match = re.search(r"edited story:\s*(.*)", response)
        edited_story = edited_story_match.group(1).strip() if edited_story_match else ""
        item["edited_story"] = edited_story
        edited_story_list.append(item)
    with open(edit_story_path, "w") as f:
        json.dump(edited_story_list, f, indent=4)

def edit_possible_story_test(
    story_path: Path,
    edit_story_path: Path,
    client: OpenAI,
):
    stories = open_jsonl(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Edit possible story in: {edit_story_path}")

    edited_story_list = []
    for item in tqdm(stories, desc="Editing possible story"):
        gold_label = item["gold_label"]
        prompt = EDIT_POSSIBLE_STORY_PROMPT.format(
            story=item["document"],
            question=item["question"],
            ending=item["options"][gold_label],
        )
        
        response = get_response(client, prompt)
        edited_story_match = re.search(r"edited story:\s*(.*)", response)
        edited_story = edited_story_match.group(1).strip() if edited_story_match else ""
        item["edited_story"] = edited_story
        del item["validation_responses"]
        del item["writer_id"]
        del item["elapsed_time_second"]
        # del item["test_responses"]
        edited_story_list.append(item)
    with open(edit_story_path, "w") as f:
        json.dump(edited_story_list, f, indent=4)

def select_story_ending(
    story_path: Path,
    ending_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Select story ending in: {ending_path}")

    story_endings = []
    for item in tqdm(stories, desc="Selecting story ending"):
        prompt = SELECT_ENDING_PROMPT.format(
            story=item["edited_story"],
            ending0=item["options"][0],
            ending1=item["options"][1],
            ending2=item["options"][2],
            ending3=item["options"][3],
        )
        response = get_response(client, prompt)
        selected_ending_match = re.search(r"selected ending:\s*(.*)", response)
        item["selected ending"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        story_endings.append(item)
    with open(ending_path, "w") as f:
        json.dump(story_endings, f, indent=4)

def select_content(
    story_path: Path,
    ending_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Select content in: {ending_path}")

    story_endings = []
    for item in tqdm(stories, desc="Selecting content"):
        prompt = SELECT_CONTENT_PROMPT.format(
            context=item["edited_story"],
            content0=item["options"][0],
            content1=item["options"][1],
            content2=item["options"][2],
            content3=item["options"][3],
        )
        response = get_response(client, prompt)
        selected_ending_match = re.search(r"selected content:\s*(.*)", response)
        item["selected content"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        story_endings.append(item)
    with open(ending_path, "w") as f:
        json.dump(story_endings, f, indent=4)

def select_content_hard(
    story_path: Path,
    ending_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Select content in: {ending_path}")

    story_endings = []
    for item in tqdm(stories, desc="Selecting content"):
        prompt = SELECT_CONTENT_PROMPT_HARD.format(
            # context=item["edited_story"],
            context=item["edited_roc_passage_with_q2"],
            question=item["question_1"],
            content0=item["options"][0],
            content1=item["options"][1],
            content2=item["options"][2],
            content3=item["options"][3],
        )
        response = get_response(client, prompt)
        selected_ending_match = re.search(r"selected content:\s*(.*)", response)
        item["selected content"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        story_endings.append(item)
    with open(ending_path, "w") as f:
        json.dump(story_endings, f, indent=4)

def test_possible_story(
    dataset_path: Path,
    result_path: Path,
    client,  
):
    dataset = open_jsonl(dataset_path)
    logger.info(f"Load dataset from: {dataset_path}")
    logger.info(f"generate result in: {result_path}")

    result_list = []
    for item in tqdm(dataset, desc="generating answer"):
        prompt = TEST_POSSIBLE_STORY_PROMPT.format(
            story=item["document"],
            question=item["question"],
            ending0=item["options"][0],
            ending1=item["options"][1],
            ending2=item["options"][2],
            ending3=item["options"][3],
        )
        response = get_response(client, prompt)
        selected_ending_match = re.search(r"selected ending:\s*(.*)", response)
        item["selected ending"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        result_list.append(item)
    with open(result_path, "w") as f:
        json.dump(result_list, f, indent=4)
        
def select_original_story_ending(
    story_path: Path,
    ending_path: Path,
    client,  # Assuming `client` is of type `OpenAI`
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Select story ending in: {ending_path}")

    story_endings = []
    response_cache: Dict[str, Dict[str, str]] = {}  # 缓存每个 roc_passage_id 的响应结果

    for item in tqdm(stories, desc="Selecting story ending"):
        roc_passage_id = item["roc_passage_id"]

        if roc_passage_id in response_cache:
            # 使用缓存的结果
            item["selected_original_ending"] = response_cache[roc_passage_id]["selected_original_ending"]
            item["original_ending_explanation"] = response_cache[roc_passage_id]["original_ending_explanation"]
            # continue
        else:
            # 构建 prompt 并调用 get_response
            prompt = SELECT_ENDING_PROMPT.format(
                story=item["document"],
                ending0=item["options"][0],
                ending1=item["options"][1],
                ending2=item["options"][2],
                ending3=item["options"][3],
            )
            response = get_response(client, prompt)

            # 提取 selected ending 和 explanation
            selected_ending_match = re.search(r"selected ending:\s*(.*)", response)
            explanation_match = re.search(r"explanation:\s*(.*)", response)

            selected_ending = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation = explanation_match.group(1).strip() if explanation_match else ""

            # 存储结果到缓存和当前 item
            response_cache[roc_passage_id] = {
                "selected_original_ending": selected_ending,
                "original_ending_explanation": explanation
            }
            item["selected_original_ending"] = selected_ending
            item["original_ending_explanation"] = explanation

        story_endings.append(item)

    with open(ending_path, "w") as f:
        json.dump(story_endings, f, indent=4)

    print(f"筛选结果已保存到 {ending_path}")

def select_reasoning_type(
    story_path: Path,
    output_path: Path,
    client: OpenAI,
):
    stories = open_jsonl(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Select reasoning type in: {output_path}")

    output_list = []
    for item in tqdm(stories, desc="Selecting reasoning type"):
        prompt = SELECT_REASONING_TYPE_PROMPT.format(
            Question=item["question"],
        )
        response = get_response(client, prompt)
        reasoning_type_match = re.search(r"(?:reasoningtype|ReasoningType):\s*(.*)", response)
        item["Type_of_Reasoning"] = reasoning_type_match.group(1).strip() if reasoning_type_match else ""

        del item["validation_responses"]
        del item["elapsed_time_second"]
        del item["test_responses"]
        output_list.append(item)
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=4)
    logger.info(f"筛选结果已保存到 {output_path}")

def edit_story_with_reasoning_type(
    story_path: Path,
    edit_story_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Edit story in: {edit_story_path}")

    edited_story_list = []
    for item in tqdm(stories, desc="Editing story"):
        if item["Type_of_Reasoning"] == "Forward Chaining" or item["Type_of_Reasoning"] == "forward chaining":
            prompt = FORWARD_CHAINING_EDIT_STORY_PROMPT.format(
                story=item["document"],
                question=item["question"],
                ending=item["options"][item["gold_label"]],
            )
        elif item["Type_of_Reasoning"] == "Backward Chaining" or item["Type_of_Reasoning"] == "backward chaining":
            prompt = BACKWARD_CHAINING_EDIT_STORY_PROMPT.format(
                story=item["document"],
                question=item["question"],
                backward_reasoning_condition=item["options"][item["gold_label"]],
            )
        
        edited_story = get_response(client, prompt)
        item["edited_story"] = re.sub(r"^Revised Story:\s*", "", edited_story, flags=re.IGNORECASE)
        edited_story_list.append(item)
    with open(edit_story_path, "w") as f:
        json.dump(edited_story_list, f, indent=4)
def generate_story_content(
    story_path: Path,
    output_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Generate story content in: {output_path}")

    output_list = []
    for item in tqdm(stories, desc="Generating story content"):
        prompt = GENERATE_STORY_CONTENT_PROMPT.format(
            context=item["edited_story"],
        )
        response = get_response(client, prompt)
        story_content_match = re.search(r"generate story content:\s*(.*)", response)
        item["generate_story_content"] = story_content_match.group(1).strip() if story_content_match else ""
        output_list.append(item)
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=4)
    logger.info(f"筛选结果已保存到 {output_path}")
def generate_inverse_option(
    story_path: Path,
    output_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Generate inverse option in: {output_path}")

    output_list = []
    for item in tqdm(stories, desc="Generating inverse option"):
        prompt = GENERATE_INVERSE_OPTION_PROMPT.format(
            context=item["document"],
            question=item["question"],
            ending=item["options"][item["gold_label"]],

        )
        response = get_response(client, prompt)
        inverse_option_match = re.search(r"inverse ending:\s*(.*)", response)
        item["inverse_option"] = inverse_option_match.group(1).strip() if inverse_option_match else ""
        output_list.append(item)
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=4)
    logger.info(f"筛选结果已保存到 {output_path}")

def detect_contradiction(
    story_path: Path,
    output_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Detect contradiction in: {output_path}")

    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Detecting contradiction"):
            prompt1 = DETECT_QUESTION_CONTRACTION_PROMPT.format(
                question1=item["question_1"],
                question2=item["question_2"],
            )
            response = get_response(client, prompt1)
            contradiction_match = re.search(r"Contradiction:\s*(.*)", response, flags=re.IGNORECASE)
            explanation_match = re.search(r"Explanation:\s*(.*)", response, flags=re.IGNORECASE)
            item["question_contradiction"] = contradiction_match.group(1).strip() if contradiction_match else ""
            item["question_contradiction_explanation"] = explanation_match.group(1).strip() if explanation_match else ""

            for i in range(4):
                prompt2 = DETECT_OPTION_CONTRACTION_PROMPT.format(
                    sentence=item["options"][i],
                )
                response = get_response(client, prompt2)
                contradiction_match = re.search(r"Contradiction:\s*(.*)", response, flags=re.IGNORECASE)
                explanation_match = re.search(r"Explanation:\s*(.*)", response, flags=re.IGNORECASE)
                item[f"option_contradiction_{i+1}"] = contradiction_match.group(1).strip() if contradiction_match else ""
                item[f"option_contradiction_explanation_{i+1}"] = explanation_match.group(1).strip() if explanation_match else ""

            # Write each item as a JSON object in the JSONL file
            f.write(json.dumps(item) + "\n")

def detect_option_issue(
    story_path: Path,
    output_path: Path,
    client: OpenAI,
):
    stories = open_json(story_path)
    logger.info(f"Load stories from: {story_path}")
    logger.info(f"Detect option issue in: {output_path}")

    data_list = []
    for item in tqdm(stories, desc="Detecting option issue"):
        prompt = DETECT_OPTION_ISSUE_PROMPT.format(
            sentence=item["options"][item["gold_label"]],
        )
        response = get_response(client, prompt)
        option_issue_match = re.search(r"issues:\s*(.*)", response)
        revision_match = re.search(r"revision:\s*(.*)", response)
        item["option_issue"] = option_issue_match.group(1).strip() if option_issue_match else ""
        item["option_issue_revision"] = revision_match.group(1).strip() if revision_match else ""
        data_list.append(item)

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)
    logger.info(f"筛选结果已保存到 {output_path}")

def edit_context(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Edit possible story in: {output_path}")

    edited_story_list = []
    for item in tqdm(stories, desc="Editing context"):
        gold_label = item["gold_label"]
        answer = re.split(r'(?<=[.!?])(?=[A-Z])', item["options"][gold_label].strip())[-1]
        prompt = EDIT_CONTEXT_PROMPT.format(
            context=item["roc_passage"],
            question=item["question_2"],
            answer=answer,
        )
        
        response = get_response(client, prompt)

        item["edited_context"] = response
        edited_story_list.append(item)
    with open(output_path, "w") as f:
        json.dump(edited_story_list, f, indent=4)

def option_optimality_verification(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
):
    stories = open_json(input_path)
    logger.info(f"Load file from: {input_path}")
    logger.info(f"Edit file in: {output_path}")

    edited_file_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="option optimality verification"):
            prompt = OPTION_OPTIMIZATION_VERIFICATION_PROMPT.format(
                context=item["edited_context"],
                question1=item["question_1"],
                question2=item["question_2"],
                options=item["options"],
            )
            
            response = get_response(client, prompt)

            contradiction_match = re.search(r"answer:\s*(.*)", response, flags=re.IGNORECASE)
            explanation_match = re.search(r"explanation:\s*(.*)", response, flags=re.IGNORECASE)
            item["option_optimality"] = contradiction_match.group(1).strip() if contradiction_match else ""
            item["option_optimality_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            
            edited_file_list.append(item)
            f.write(json.dumps(item) + "\n")

    with open(output_path, "w") as f:
        json.dump(edited_file_list, f, indent=4)

def select_content_with_implicit_explicit_cot_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_COT_PROMPT.format(
                context=item["edited_context"],
                question=item["question_1"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            response = get_response(client, prompt, model_name)
            selected_ending_match = re.search(r"answer:\s*(.*)", response)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation_match = re.search(r"explanation:\s*(.*)", response)
            item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            item["response"] = response
            data_list.append(item)
            time.sleep(3)
            f.write(json.dumps(item) + "\n")

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)


def dashscope_select_content_with_implicit_explicit_cot_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    # with open(output_path, "a") as f:
    for item in tqdm(stories, desc="Selecting content"):
        option_order = item["shuffled_indices"]
        prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_COT_PROMPT.format(
            context=item["edited_context"],
            question=item["question_1"],
            option0=item["options"][0],
            option1=item["options"][1],
            option2=item["options"][2],
            option3=item["options"][3],
        )
        response = get_response(client, prompt, model_name)
        selected_ending_match = re.search(r"answer:\s*(.*)", response)
        item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        item["response"] = response
        data_list.append(item)
        time.sleep(3)
            # f.write(json.dumps(item) + "\n")

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)


def select_content_with_implicit_explicit_one_shot_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    # with open(output_path, "a") as f:
    for item in tqdm(stories, desc="Selecting content"):
        option_order = item["shuffled_indices"]
        prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_ONE_SHOT_PROMPT.format(
            context=item["edited_context"],
            question=item["question_1"],
            option0=item["options"][0],
            option1=item["options"][1],
            option2=item["options"][2],
            option3=item["options"][3],
        )
        response = get_response(client, prompt, model_name)
        selected_ending_match = re.search(r"answer:\s*(.*)", response)
        item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        data_list.append(item)
        time.sleep(4)
            # f.write(json.dumps(item) + "\n")

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)

def select_content_with_implicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_IMPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            response = get_response(client, prompt, model_name)
            selected_ending_match = re.search(r"answer:\s*(.*)", response)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation_match = re.search(r"explanation:\s*(.*)", response)
            item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            data_list.append(item)
            time.sleep(3)
            f.write(json.dumps(item) + "\n")

    # with open(output_path, "w") as f:
    #     json.dump(data_list, f, indent=4)

def select_content_with_implicit_explicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                question=item["question_1"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            response = get_response(client, prompt, model_name)
            selected_ending_match = re.search(r"answer:\s*(.*)", response)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation_match = re.search(r"explanation:\s*(.*)", response)
            item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            data_list.append(item)
            # time.sleep(3)
            f.write(json.dumps(item) + "\n")

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)

def o1_preview_select_content_with_implicit_explicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                question=item["question_1"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            temperature = 0
            response = get_response(client, prompt, model_name, temperature)
            selected_ending_match = re.search(r"answer:\s*(.*)", response)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation_match = re.search(r"explanation:\s*(.*)", response)
            item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            data_list.append(item)
            f.write(json.dumps(item) + "\n")

    # with open(output_path, "w") as f:
    #     json.dump(data_list, f, indent=4)
def select_content_with_explicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_EXPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                question=item["question_2"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            response = get_response(client, prompt, model_name)
            selected_ending_match = re.search(r"answer:\s*(.*)", response)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation_match = re.search(r"explanation:\s*(.*)", response)
            item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            data_list.append(item)
            f.write(json.dumps(item) + "\n")
            # time.sleep(4)

    # with open(output_path, "w") as f:
    #     json.dump(data_list, f, indent=4)

def select_content_with_double_explicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    for item in tqdm(stories, desc="Selecting content"):
        # option_order = item["shuffled_indices"]
        prompt = SELECT_CONTENT_WITH_DOUBLE_EXPLICIT_SETTING_PROMPT.format(
            context=item["edited_context"],
            question1=item["question_1"],
            question2=item["question_2"],
            option0=item["options"][0],
            option1=item["options"][1],
            option2=item["options"][2],
            option3=item["options"][3],
        )
        response = get_response(client, prompt, model_name)
        selected_ending_match = re.search(r"answer:\s*(.*)", response)
        item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
        explanation_match = re.search(r"explanation:\s*(.*)", response)
        item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        data_list.append(item)
        time.sleep(3)

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)
def o1_preview_select_content_with_double_explicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    # model_name: str,
):

    model_name = "gpt-4o-mini-2024-07-18"
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            # option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_DOUBLE_EXPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                question1=item["question_1"],
                question2=item["question_2"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            temperature = 1
            response = get_response(client, prompt, model_name, temperature)
            selected_ending_match = re.search(r"answer:\s*(.*)", response)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation_match = re.search(r"explanation:\s*(.*)", response)
            item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            # data_list.append(item)

            f.write(json.dumps(item) + "\n")

    # with open(output_path, "w") as f:
    #     json.dump(data_list, f, indent=4)

def test_r1_select_content_with_implicit_explicit_setting(
    input_path: Path,
    output_path: Path,
    # client: OpenAI,
    # model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    client = OpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                question=item["question_1"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            completion = client.chat.completions.create(
                model="deepseek-r1", 
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                # 解除以下注释会在最后一个chunk返回Token使用量
                # stream_options={
                #     "include_usage": True
                # }
            )
            # 定义完整思考过程
            reasoning_content = ""
            # 定义完整回复
            answer_content = ""
            # 判断是否结束思考过程并开始回复
            is_answering = False

            # print("\n"+"="*20+"思考过程"+"="*20+"\n")
            for chunk in completion:
                # include_usage 设置为 True 会使得最后一个chunk返回 Token 使用量，而choices为空列表，此处进行判断
                if chunk.choices == []:
                    # print("\n"+"="*20+"Token 使用情况"+"="*20+"\n")
                    # print(chunk.usage)
                    pass
                # 以下为思考与回复的步骤
                else:
                    # include_usage 设置为 True 时，倒数第二个chunk会不包含 reasoning_content 字段，因此需要进行判断
                    if hasattr(chunk.choices[0].delta, 'reasoning_content') == False:
                        pass
                    else:
                        # 有时可能会出现思考过程与回复皆为空的情况，此时忽略即可
                        if chunk.choices[0].delta.reasoning_content == "" and chunk.choices[0].delta.content == "":
                            pass
                        else:
                            # 如果思考结果为空，则开始打印完整回复
                            if chunk.choices[0].delta.reasoning_content == "" and is_answering == False:
                                # print("\n"+"="*20+"完整回复"+"="*20+"\n")
                                # 防止打印多个“完整回复”标记
                                is_answering = True
                            # 如果思考过程不为空，则打印思考过程
                            if chunk.choices[0].delta.reasoning_content != "":
                                # print(chunk.choices[0].delta.reasoning_content,end="")
                                reasoning_content += chunk.choices[0].delta.reasoning_content
                            # 如果回复不为空，则打印回复。回复一般会在思考过程结束后返回
                            elif chunk.choices[0].delta.content != "":
                                # print(chunk.choices[0].delta.content,end="")
                                answer_content += chunk.choices[0].delta.content


            selected_ending_match = re.search(r"answer:\s*(.*)", answer_content)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
            explanation_match = re.search(r"explanation:\s*(.*)", answer_content)
            item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            # data_list.append(item)
            f.write(json.dumps(item) + "\n")


    # with open(output_path, "w") as f:
    #     json.dump(data_list, f, indent=4)


def test_math_select_content_with_implicit_explicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                question=item["question_1"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            response = get_response(client, prompt, model_name)
            selected_ending_match = re.search(r"\\boxed{([^}]*)}", response)
            item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""

            item["explanation"] = response
            data_list.append(item)
            time.sleep(3)
            f.write(json.dumps(item) + "\n")
    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)


def dashscope_select_content_with_implicit_explicit_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):
            option_order = item["shuffled_indices"]
            prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT.format(
                context=item["edited_context"],
                question=item["question_1"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            completion = dashscope.Generation.call(
                api_key=os.getenv("ALIYUN_API_KEY"),
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                result_format='message',  # set the result to be "message" format.
                temperature=0.1
            )
            if completion.status_code == HTTPStatus.OK:
                # print("completion:\n",completion)
                response = completion.output.choices[0].message.content
                # response = get_response(client, prompt, model_name)
                # selected_ending_match = re.search(r"answer:\s*(.*)", response)
                selected_ending_match = re.search(r"answer:\s*(.*)", response)
                item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
                explanation_match = re.search(r"explanation:\s*(.*)", response)
                item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
                item["response"] = response
                data_list.append(item)
                f.write(json.dumps(item) + "\n")
                # time.sleep(2)
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    completion.request_id, completion.status_code,
                    completion.code, completion.message
                ))

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)


def dashscope_select_content_with_implicit_explicit_cot_setting(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    model_name: str,
):
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"Select content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="Selecting content"):

            prompt = SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_COT_PROMPT.format(
                context=item["edited_context"],
                question=item["question_1"],
                option0=item["options"][0],
                option1=item["options"][1],
                option2=item["options"][2],
                option3=item["options"][3],
            )
            completion = dashscope.Generation.call(
                api_key=os.getenv("ALIYUN_API_KEY"),
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                result_format='message',  # set the result to be "message" format.
                temperature=0.1
            )
            if completion.status_code == HTTPStatus.OK:
                # print("completion:\n",completion)
                response = completion.output.choices[0].message.content
                # response = get_response(client, prompt, model_name)
                # selected_ending_match = re.search(r"answer:\s*(.*)", response)
                selected_ending_match = re.search(r"answer:\s*(.*)", response)
                item["answer"] = selected_ending_match.group(1).strip() if selected_ending_match else ""
                explanation_match = re.search(r"explanation:\s*(.*)", response)
                item["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
                item["response"] = response
                data_list.append(item)
                f.write(json.dumps(item) + "\n")
                # time.sleep(2)
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    completion.request_id, completion.status_code,
                    completion.code, completion.message
                ))

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)

def main(
    module_dir: Path,
    step: str,
    max_backgrounds_per_theme: int = 30,
    max_personas_per_background: int = 9,
    max_events_per_persona: int = 2,
    max_endings_per_event: int = 1,
    max_twists_per_ending: int = 1,
    max_dialogues_per_emotion: int = 3,
    max_story_num: int = 10,
    # model_name: str,
):

    match step:
        # case "background":
        #     collect_background(
        #         Path(os.path.join(module_dir,"theme.json")),
        #         Path(os.path.join(module_dir,"background_qwen2.5_70B.json")),
        #         # module_dir / "theme.json",
        #         # module_dir / "background_qwen.json",
        #         client,
        #         max_backgrounds_per_theme,
        #     )
        # case "persona":
        #     collect_persona(
        #         Path(os.path.join(module_dir,"background_qwen2.5_70B.json")),
        #         Path(os.path.join(module_dir,"persona_qwen2.5_70B.json")),                
        #         # module_dir / "background.json",
        #         # module_dir / "persona.json",
        #         client,
        #         max_personas_per_background,
        #     )
        # case "event":
        #     collect_event(
        #         Path(os.path.join(module_dir,"persona_qwen2.5_70B.json")),
        #         Path(os.path.join(module_dir,"event_qwen2.5_70B.json")),
        #         # module_dir / "persona.json",
        #         # module_dir / "event.json",
        #         client,
        #         max_events_per_persona,
        #     )
        # case "ending":
        #     collect_ending(
        #         Path(os.path.join(module_dir,"event_qwen2.5_70B.json")),
        #         Path(os.path.join(module_dir,"ending_qwen2.5_70B.json")),                
        #         # module_dir / "event.json",
        #         # module_dir / "ending.json",
        #         client,
        #         max_endings_per_event,
        #     )
        # case "twist":
        #     collect_twist(
        #         Path(os.path.join(module_dir,"ending_qwen2.5_70B.json")),
        #         Path(os.path.join(module_dir,"twist_qwen_qwen2.5_70B.json")),                  
        #         # module_dir / "ending.json",
        #         # module_dir / "twist.json",
        #         client,
        #         max_twists_per_ending,
        #     )
        # case "dialogue":
        #     generate_dialogue(
        #         Path(os.path.join(module_dir,"dialogue_llama3.1_70B_1k.json")),
        #         client,
        #         max_dialogues_per_emotion,
        #     )
        # case "story_beginning":
        #     generate_story_beginning(
        #         Path(os.path.join(module_dir,"dialogue_llama3.1_70B_1k.json")),
        #         Path(os.path.join(module_dir,"story_beginning_llama3.1_70B_1k.json")),
        #         client,
        #     )
        # case "story_ending":
        #     generate_story_ending(
        #         Path(os.path.join(module_dir,"story_beginning_llama3.1_70B.json")),
        #         Path(os.path.join(module_dir,"story_ending_llama3.1_70B.json")),
        #         client,
        #     )
        # case "revise_emotion":
        #     revise_emotion(
        #         Path(os.path.join(module_dir,"story_beginning_llama3.1_70B_1k.json")),
        #         Path(os.path.join(module_dir,"revise_emotion_llama3.1_70B_1k.json")),
        #         client,
        #     )
        # case "calculate_similarity":
        #     calculate_similarity(
        #         Path(os.path.join(module_dir,"revise_ending_llama3.1_70B.json")),
        #         Path(os.path.join(module_dir,"calculate_similarity_llama3.1_70B.json")),
        #         client,
        #     )
        # case "five_sentences_story":
        #     generate_5_sentences_story(
        #         Path(os.path.join(module_dir,"five_sentences_story_llama3.1_70B_v2.json")),
        #         client,
        #         max_story_num,
        #     )
        
        # case "edit_story":
        #     edit_story_v2(
        #         Path(os.path.join(module_dir,"story_mops_llama3_1_five_sentences_beginning.jsonl")),
        #         Path(os.path.join(module_dir,"edit_five_sentences_beginning_llama3.1_70B_mops.json")),
        #         client,
        #     )
        # case "edit_story_with_element":
        #     edit_story_with_element(
        #         Path(os.path.join(module_dir,"edit_five_sentences_beginning_llama3.1_70B_mops.json")),
        #         Path(os.path.join(module_dir,"edited_five_sentences_beginning_llama3.1_70B_mops_with_element.json")),
        #         client,
        #     )
        # case "generate_one_sentence_ending":
        #     generate_one_sentence_ending(
        #         # Path(os.path.join(module_dir,"edit_five_sentences_story_llama3.1_70B_v2.json")),
        #         Path(os.path.join(module_dir,"edited_five_sentences_story_llama3.1_70B_with_element_v2.json")),
        #         Path(os.path.join(module_dir,"one_sentence_ending_qwen2.5_72B_mops.json")),
        #         client,
        #     )
        # case "generate_edited_story_ending":
        #     generate_edited_story_ending(
        #         # Path(os.path.join(module_dir,"edited_five_sentences_beginning_llama3.1_70B_mops_with_element.json")),
        #         Path(os.path.join(module_dir,"edit_five_sentences_beginning_setting_llama3.1_70B_mops.json")),
        #         # Path(os.path.join(module_dir,"edited_story_ending_llama3.1_70B_mops.json")),
        #         Path(os.path.join(module_dir,"edit_ending_setting_llama3.1_70B_mops.json")),
        #         client,
        #     )
        # case "generate_original_story_ending":
        #     generate_original_story_ending(
        #         # Path(os.path.join(module_dir,"edited_story_ending_llama3.1_70B_mops.json")),
        #         Path(os.path.join(module_dir,"edit_ending_setting_llama3.1_70B_mops.json")),
        #         # Path(os.path.join(module_dir,"full_story_llama3.1_70B_mops.json")),
        #         Path(os.path.join(module_dir,"full_story_edit_setting_llama3.1_70B_mops.json")),
        #         client,
        #     )
        # case "extract_story_elements":
        #     extract_story_elements(
        #         Path(os.path.join(module_dir,"full_story_llama3.1_70B_mops.json")),
        #         Path(os.path.join(module_dir,"extract_story_elements_llama3.1_70B_mops.json")),
        #         # Path(os.path.join(module_dir,"full_story_llama3.1_70B_mops_test.json")),
        #         # Path(os.path.join(module_dir,"extract_story_elements_llama3.1_70B_mops_test.json")),
        #         client,
        #     )
        # case "calculate_element_similarity":
        #     calculate_element_similarity(
        #         Path(os.path.join(module_dir,"extract_story_elements_llama3.1_70B_mops.json")),
        #         Path(os.path.join(module_dir,"calculate_element_similarity_llama3.1_70B_mops.json")),
        #         client,
        #     )
        # case "edit_setting":
        #     edit_setting(
        #         Path(os.path.join(module_dir,"story_mops_llama3_1_five_sentences_beginning.jsonl")),
        #         # Path(os.path.join(module_dir,"test.jsonl")),
        #         Path(os.path.join(module_dir,"edit_five_sentences_beginning_setting_llama3.1_70B_mops.json")),
        #         client,
        #     )
        # case "edit_possible_story":
        #     edit_possible_story(
        #         Path("/data/code/possible-stories/dataset/train.jsonl"),
        #         # Path(os.path.join(module_dir,"edit_possible_story_llama3.1_70B_mops.json")),
        #         # Path(os.path.join(module_dir,"edit_possible_story_llama3.1_70B_new.json")),
        #         Path(os.path.join(module_dir,"edit_possible_story_qwen2.5_72B_new.json")),
        #         client,
        #     )
        case "select_story_ending":
            select_story_ending(
                # Path(os.path.join(module_dir,"edit_possible_story_llama3.1_70B_new.json")),
                # Path(os.path.join(module_dir,"edit_possible_story_qwen2.5_72B_new.json")),
                # Path(os.path.join(module_dir,"test.json")),

                Path(os.path.join(module_dir,"edited_possible_story_test_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"edited_possible_story_test_llama3.1_70B.json")),

                # Path(os.path.join(module_dir,"edited_possible_story_train_llama3.1_70B.json")),
                # Path(os.path.join(module_dir,"edited_possible_story_train_qwen2.5_72B.json")),
                
                


                # Path(os.path.join(module_dir,"select_possible_story_ending_llama3.1_70B_mops.json")),
                # Path(os.path.join(module_dir,"select_possible_story_ending_qwen2.5_72B_mops.json")),
                # Path(os.path.join(module_dir,"select_possible_story_ending_qwen2.5_14B.json")),
                # Path(os.path.join(module_dir,"select_possible_story_ending_qwen2.5_72B_new.json")),
                # Path(os.path.join(module_dir,"select_possible_story_ending_on_qwendata_qwen2.5_72B_new.json")),

                # Path(os.path.join(module_dir,"qwen2.5_edited_test_llama3.1_70B.json")),
                # Path(os.path.join(module_dir,"llama3.1_edited_test_llama3.1_70B.json")),

                # Path(os.path.join(module_dir,"qwen2.5_edited_test_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"llama3.1_edited_test_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_test_qwen2.5_14B.json")),
                # Path(os.path.join(module_dir,"llama3.1_edited_test_qwen2.5_14B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_test_qwen2.5_7B.json")),
                # Path(os.path.join(module_dir,"llama3.1_edited_test_qwen2.5_7B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_test_gpt-4o-2024-08-06.json")),
                Path(os.path.join(module_dir,"qwen2.5_edited_test_gpt-4-0125-preview.json")),

                

                # Path(os.path.join(module_dir,"llama3.1_edited_train_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_train_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_train_llama3.1_70B.json")),
                # Path(os.path.join(module_dir,"llama3.1_edited_train_llama3.1_70B.json")),
                

                # Path(os.path.join(module_dir,"test_llama3.1_70B.json")),
                # Path(os.path.join(module_dir,"test_qwen2.5_14B.json")),
                # Path(os.path.join(module_dir,"test_qwen2.5_7B.json")),
                client,
            )
        # case "select_original_story_ending":
        #     select_original_story_ending(
        #         # Path(os.path.join(module_dir,"select_possible_story_ending_llama3.1_70B_mops.json")),
        #         Path(os.path.join(module_dir,"select_possible_story_ending_qwen2.5_72B_mops.json")),
        #         # Path(os.path.join(module_dir,"select_possible_story_ending_qwen2.5_14B.json")),
        #         # Path(os.path.join(module_dir,"possible_story_ending_llama3.1_70B_mops.json")),
        #         # Path(os.path.join(module_dir,"possible_story_ending_qwen2.5_72B_mops.json")),
        #         Path(os.path.join(module_dir,"possible_story_ending_qwen2.5_14B.json")),
        #         client,
        #     )
        # case "edit_possible_story_test":
        #     edit_possible_story_test(
        #         # Path("/data/code/possible-stories/dataset/test.jsonl"),
        #         Path("/data/code/possible-stories/dataset/train.jsonl"),
        #         Path("/data/code/MoPS/assets/PS_train/edited_possible_story_train_llama3.1_70B.json"),
        #         # Path("/data/code/MoPS/assets/PS_train/edited_possible_story_train_qwen2.5_72B.json"),
        #         client,
        #     )
        case "test_possible_story":
            test_possible_story(
                Path("/data/code/possible-stories/dataset/test.jsonl"),
                # Path("/data/code/MoPS/assets/PS_test/llama3.1_70B_on_PS_test.json"),
                # Path("/data/code/MoPS/assets/PS_test/qwen2.5_72B_on_PS_test.json"),
                # Path("/data/code/MoPS/assets/PS_test/qwen2.5_14B_on_PS_test.json"),
                # Path("/data/code/MoPS/assets/PS_test/qwen2.5_7B_on_PS_test.json"),
                # Path("/data/code/MoPS/assets/PS_test/gpt-4o-2024-08-06_on_PS_test.json"),
                Path("/data/code/MoPS/assets/PS_test/llama3.1_8B_on_PS_test.json"),
                client, 
            )
        # case "select_reasoning_type":
        #     select_reasoning_type(
        #         Path("/data/code/possible-stories/dataset/test.jsonl"),
        #         Path("/data/code/MoPS/assets/PS_test/test_with_ReasoningType_qwen2.5.json"),
        #         client, 
        #     )
        # case "edit_story_with_reasoning_type":
        #     edit_story_with_reasoning_type(
        #         Path("/data/code/MoPS/assets/PS_test/test_with_ReasoningType_qwen2.5.json"),
        #         Path("/data/code/MoPS/assets/PS_test/edit_story_with_reasoning_type_qwen2.5_v2.json"),
        #         client, 
        #     )
        # case "select_content":
        #     select_content(
        #         Path(os.path.join(module_dir,"edit_story_with_reasoning_type_qwen2.5_v2.json")),
        #         # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_llama3.1.json")),
        #         # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_qwen2.5_72B.json")),
        #         # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_gpt4o.json")),
        #         # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_qwen2.5_14B.json")),
        #         # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_qwen2.5_7B.json")),
        #         # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_llama3.1_8B.json")),
        #         Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_llama3.2_3B.json")),
        #         client,
        #     ) 
        # case "generate_story_content":
        #     generate_story_content(
        #         Path(os.path.join(module_dir,"edit_story_with_reasoning_type_qwen2.5_v2.json")),
        #         Path(os.path.join(module_dir,"qwen2.5_generate_story_content_test_qwen2.5_72B.json")),
        #         client,
        #     )
        case "select_content_hard":
            select_content_hard(
                # Path(os.path.join(module_dir,"hard_dataset.json")),
                # Path(os.path.join(module_dir,"mixed_question_gpt4o.json")),
                Path(os.path.join(module_dir,"revised_mixed_question.json")),


                # Path(os.path.join(module_dir,"hard_dataset_res_llama3.1_70B.json")),
                # Path(os.path.join(module_dir,"hard_dataset_res_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_gpt4o.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_qwen2.5_14B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_qwen2.5_7B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_llama3.1_8B.json")),
                # Path(os.path.join(module_dir,"qwen2.5_edited_with_reasoning_type_test_llama3.2_3B.json")),

                # Path(os.path.join(module_dir,"mixed_question_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"mixed_question_gpt4o.json")),
                # Path(os.path.join(module_dir,"mixed_question_llama3.1_70B.json")),
                # Path(os.path.join(module_dir,"mixed_question_qwen2.5_14B.json")),
                # Path(os.path.join(module_dir,"mixed_question_qwen2.5_7B.json")),
                # Path(os.path.join(module_dir,"mixed_question_llama3.1_8B.json")),


                # Path(os.path.join(module_dir,"revised_mixed_question_gpt4o.json")),
                # Path(os.path.join(module_dir,"revised_mixed_question_qwen2.5_72B.json")),
                # Path(os.path.join(module_dir,"revised_mixed_question_llama3.1_70B.json")),
                # Path(os.path.join(module_dir,"revised_mixed_question_qwen2.5_14B.json")),
                # Path(os.path.join(module_dir,"revised_mixed_question_qwen2.5_7B.json")),
                Path(os.path.join(module_dir,"revised_mixed_question_llama3.1_8B.json")),
                client,
            )         
        case "generate_inverse_option":
            generate_inverse_option(
                Path(os.path.join(module_dir,"edit_story_with_reasoning_type_qwen2.5_v2.json")),
                Path(os.path.join(module_dir,"edit_story_with_inverse_option_gpt4o.json")),
                client,
            )
        case "detect_contradiction":
            detect_contradiction(
                # Path(os.path.join(module_dir,"mixed_question_gpt4o.json")),
                Path(os.path.join(module_dir,"mixed_question_gpt4o_copy.json")),
                
                Path(os.path.join(module_dir,"contradiction_mixed_question_gpt4o.jsonl")),
                client,
            )
        case "detect_option_issue":
            detect_option_issue(
                Path(os.path.join(module_dir,"revised_mixed_question.json")),
                Path(os.path.join(module_dir,"option_issue_mixed_question_llama3.1.json")),
                client,
            )
        case "edit_context":
            edit_context(
                Path(os.path.join(module_dir,"revised_mixed_question.json")),
                Path(os.path.join(module_dir,"edited_mixed_question.json")),
                client,
            )
        case "option_optimality_verification":
            option_optimality_verification(
                # Path(os.path.join(module_dir,"edited_mixed_question_new_version.json")),
                # Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options.json")),
                Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final.json")),
                # Path(os.path.join(module_dir,"option_optimality_gpt4o_3-31.json")),
                # Path(os.path.join(module_dir,"option_optimality_deepseek-v3_3-31.json")),
                # Path(os.path.join(module_dir,"option_optimality_qwen-max-2025-01-25_3-31.json")),

                # Path(os.path.join(module_dir,"option_optimality_gpt4o_3-31_v2.json")),
                # Path(os.path.join(module_dir,"option_optimality_deepseek-v3_3-31_v2.json")),
                Path(os.path.join(module_dir,"option_optimality_qwen-max-2025-01-25_3-31_v2.json")),
                client,
            )
        case "select_content_with_implicit_explicit_setting":
            select_content_with_implicit_explicit_setting(
                Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options.json")),
                
                # Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options_gpt4o_temp0.json")),
                # Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options_llama3.1_70b.jsonl")),
                # Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options_Qwen2.5-72B-Instruct-AWQ_temp0.json")),
                # Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options_Qwen2.5-14B-Instruct-AWQ_temp0.json")),
                # Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options_Qwen2.5-7B-Instruct-AWQ_temp0.json")),
                Path(os.path.join(module_dir,"after_option_optimality_mixed_question_final_shuffled_options_Llama-3.1-8B-Instruct_temp0.json")),
                client,
                model_name,
            )

        case "test_r1":
            test_r1_select_content_with_implicit_explicit_setting(
                Path("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options.json"),
                Path("/data/code/MoPS/assets/explicit_implicit_setting_new/deepseek-r1.jsonl")
            )
        case "test_math":
            test_math_select_content_with_implicit_explicit_setting(
                Path("/data/code/MoPS/assets/test_r1.json"),
                Path("/data/code/MoPS/assets/test_math_answer.json")                
            )
        case "o1_double_ex":
            o1_preview_select_content_with_double_explicit_setting(
                Path("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options_o1_double_ex.json"),
                Path("/data/code/MoPS/assets/double_explicit_setting_new/o1-preview-2024-09-12.jsonl")
            )






if __name__ == "__main__":
    # tyro.cli(main)
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--module-dir', type=str, help='Path to the module')
    argparse.add_argument('--step', type=str, help='process step')
    argparse.add_argument('--max-backgrounds-per-theme', type=int, default=30)
    argparse.add_argument('--max-personas-per-background', type=int, default=9)
    argparse.add_argument('--max-events-per-persona', type=int, default=2)
    argparse.add_argument('--max-endings-per-event', type=int, default=1)
    argparse.add_argument('--max-twists-per-ending', type=int, default=1)
    argparse.add_argument('--max_dialogues_per_emotion', type=int, default=3)
    argparse.add_argument('--max_story_num', type=int, default=10)
    argparse.add_argument('--model_name', type=str,  help='model_name')


    args = argparse.parse_args()
    main(args.module_dir, args.step, args.max_backgrounds_per_theme, args.max_personas_per_background,
    args.max_events_per_persona,args.max_endings_per_event, args.max_twists_per_ending,args.max_dialogues_per_emotion,
    args.max_story_num,
    # args.model_name
    )

    # cal_bertscore_v2(
    #     Path(os.path.join("/data/code/MoPS/assets/mops_five_sent","one_sentence_ending_qwen2.5_72B_mops.json")),
    #     Path(os.path.join("/data/code/MoPS/assets/mops_five_sent","one_sentence_ending_qwen2.5_72B_mops_bertscore.json")),)