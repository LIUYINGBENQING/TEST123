import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from collections import defaultdict,Counter
from mops.constants import logger
from tqdm import tqdm
import random
from openai import OpenAI
from mops.logger import get_logger
from mops.prompts import TRANSLATE_PROMPT
import re
import csv
import os



def embedding(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> NDArray:
    model = SentenceTransformer(model_name)
    embedding = model.encode(texts)
    assert isinstance(embedding, np.ndarray)
    return embedding


def dim_reduction(
    embeddings: NDArray, perplexity: int = 50, random_state: int = 42
) -> NDArray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    vis_dims = tsne.fit_transform(embeddings)
    assert vis_dims.shape == (embeddings.shape[0], 2)
    return vis_dims


def open_json(file: Path, mode: str = "r", create_if_not_exists: bool = False):
    if not file.exists() and create_if_not_exists:
        file.write_text(json.dumps({}))
        logger.warning(f"Create json file at: {file}")

    with open(file, mode) as fp:
        data = json.load(fp)
    return data


def open_jsonl(file: Path, mode: str = "r", create_if_not_exists: bool = True):
    if not file.exists() and create_if_not_exists:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()  # Creates an empty file
        logger.info(f"Create jsonl file at: {file}")

    with open(file, mode, encoding="utf-8") as fp:
        data = [json.loads(line) for line in fp.readlines()]
        return data


def save_json(json_dict: Dict, file: Path, mode: str = "w"):
    with open(file, mode, encoding="utf-8") as fp:
        json.dump(json_dict, fp, ensure_ascii=False, indent=4)


def save_jsonl(
    json_lines: List[Dict],
    file: Path,
    mode: str = "w",
    ensure_ascii=True,
):
    with open(file, mode, encoding="utf-8") as fp:
        for json_line in json_lines:
            fp.write(json.dumps(json_line, ensure_ascii=ensure_ascii) + "\n")


def save_fig(fig: Figure, fig_path: Path, **kwargs):
    if not fig_path.exists():
        fig_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Save figure to: {fig_path}")
    fig.savefig(fig_path, **kwargs)

def post_process_story_ending(story_ending:str):
    import re
    pattern = r"### (Completed Story|Story Continuation)\n\n([\s\S]*?)(?=\n###|\Z)"
    matches = re.findall(pattern, story_ending)
    if matches:
        return matches[0][1]
    else:
        return story_ending

def add_question(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    data_list = []
    for item in data:
        item["new_question"] = "Which ending best fits the story beginning in terms of plot, tone, or logical progression?"
        data_list.append(item)
    with open(output_file, 'w') as f:
        for entry in data_list:
            json.dump(entry, f)
            f.write('\n')  

def postprocess_edited_story(input_file, output_file):
    import re
    with open(input_file, 'r') as f:
        data = json.load(f)
    data_list = []
    for item in data:
        item["edited_story"] = re.sub(r"^Revised Story:\s*", "", item["edited_story"], flags=re.IGNORECASE)
        data_list.append(item)
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=4)

    return text
    
def collect_question(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    result = defaultdict(list)
    for entry in data:
        roc_passage_id = entry.get("roc_passage_id")
        question_id = entry.get("question_id")
        question_text = entry.get("question")
        option_text = entry["options"][entry["gold_label"]]
        inverse_option = entry.get("inverse_option")
        
        # 将问题和 inverse_option 存储到 result 字典中，按照 roc_passage_id 汇总
        result[roc_passage_id].append({
            "question_id": question_id,
            "question": question_text,
            "document": entry.get("document"),
            "edited_context": entry.get("edited_context"),
            "options": entry.get("options"),
            "gold_label": entry.get("gold_label"),
            "inverse_option": inverse_option
        })
    with open(output_file, 'w',) as f:
        json.dump(result, f, indent=4)
        
def generate_mixed_questions(all_data):
    mixed_question_num = 0  # 用于生成混合问题的唯一ID
    mixed_questions = []  # 存储所有混合问题

    # 遍历所有的roc_passage_id
    for roc_passage_id, question_set in all_data.items():
        # 遍历问题集合，生成问题组合
        for question_1 in question_set:
            for question_2 in question_set:
                # 排除自己和相似问题（gold_label相同）
                if question_1["gold_label"] == question_2["gold_label"]:
                    continue

                

                # 生成混合问题
                mixed_question = {}
                mixed_question["mixed_question_id"] = mixed_question_num
                mixed_question["roc_passage_id"] = roc_passage_id
                mixed_question["roc_passage"] = question_1["document"]
                mixed_question["edited_roc_passage_with_q2"] = question_2["edited_context"]
                mixed_question["question_1_id"] = question_1["question_id"]
                mixed_question["question_1"] = question_1["question"]
                mixed_question["question_2_id"] = question_2["question_id"]
                mixed_question["question_2"] = question_2["question"]

                # 生成问题选项
                # 获取 question_1 和 question_2 的 options 和 inverse_option
                option_1 = question_1["options"]
                inverse_option_1 = question_1["inverse_option"]
                option_2 = question_2["options"]
                inverse_option_2 = question_2["inverse_option"]

                # 组合问题选项 1+2、1+-2、-1+2
                mixed_option_1 = option_1[question_1["gold_label"]] + option_2[question_2["gold_label"]]
                mixed_option_2 = option_1[question_1["gold_label"]] + inverse_option_2
                mixed_option_3 = inverse_option_1 + option_2[question_2["gold_label"]]

                # 处理选项 1+3
                if question_1["gold_label"] != 0 and question_2["gold_label"] != 0:
                    irrelevant_option_num = 0
                elif question_1["gold_label"] != 1 and question_2["gold_label"] != 1:
                    irrelevant_option_num = 1
                elif question_1["gold_label"] != 2 and question_2["gold_label"] != 2:
                    irrelevant_option_num = 2
                else:
                    irrelevant_option_num = 3
                mixed_option_4 = option_1[question_1["gold_label"]] + option_1[irrelevant_option_num]

                # 将所有选项存储到混合问题中
                mixed_question["options"] = [mixed_option_1, mixed_option_2, mixed_option_3, mixed_option_4]
                mixed_question["gold_label"] = 0  # 默认将 gold_label 设置为0

                # 添加混合问题到列表
                mixed_questions.append(mixed_question)
                mixed_question_num += 1  # 增加混合问题计数

    return mixed_questions

def process_json(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        all_data = json.load(infile)
    
    # 生成混合问题
    mixed_questions = generate_mixed_questions(all_data)
    
    # 将生成的混合问题写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(mixed_questions, outfile, ensure_ascii=False, indent=4)

def count_answer_distribution(input_file):
    # 初始化计数器
    distribution = {"0": 0, "1": 0, "2": 0, "3": 0}
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        all_data = json.load(infile)
    
    for item in all_data:

        selected_content = item.get("selected content", None)
        if selected_content is not None and selected_content in distribution:
            distribution[selected_content] += 1

    return distribution

def jsonl_to_json(jsonl_path: Path, json_path: Path):
    # Open the JSONL file and the output JSON file
    with open(jsonl_path, "r") as jsonl_file, open(json_path, "w") as json_file:
        json_lines = jsonl_file.readlines()
        json_data = [json.loads(line) for line in json_lines]
        json.dump(json_data, json_file,indent=4)

    print(f"Converted {jsonl_path} to {json_path}")

def del_contradict_item(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    data_list = []
    mixed_question_num = 0
    for item in data:
        if item["question_contradiction"] == "true":
            continue
        item["mixed_question_id"] = mixed_question_num
        mixed_question_num += 1
        data_list.append(item)
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=4)

def count_contradiction(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    counts = {"1_true": 0, "2_true": 0, "3_true": 0}
    one_true_count = []
    two_true_count = []
    three_true_count = []

    for item in data:
        true_count = sum(
            1 for key in ["option_contradiction_2", "option_contradiction_3", "option_contradiction_4"]
            if item.get(key) == "true"
        )
        if true_count == 1:
            counts["1_true"] += 1
            one_true_count.append(item)
        elif true_count == 2:
            counts["2_true"] += 1
            two_true_count.append(item)
        elif true_count == 3:
            counts["3_true"] += 1
            three_true_count.append(item)

    return counts

def count_option_optimality(input_path: Path):

    with open(input_path, "r") as f:
        data = json.load(f)

    # data_list = defaultdict(int)
    num = 0
    for item in tqdm(data, desc="count option optimality"):
        key = item.get("option_optimality")  
        if key in { "1", "2", "3"}: 
            # data_list[key] += 1
            num += 1
            if num == 123:
                print("123:", item["mixed_question_id"])


    
    # print(data_list)


def reorder_json_file(input_file, output_file, key_order):
    with open(input_file, "r", encoding="utf-8") as infile:
        json_data = json.load(infile)
    
    ordered_data = []
    for item in json_data:
        # 按指定顺序排列键
        ordered_item = {key: item[key] for key in key_order if key in item}
        # 将未指定的键添加到末尾
        unordered_keys = {key: item[key] for key in item if key not in key_order}
        ordered_item.update(unordered_keys)

        ordered_data.append(ordered_item)
    # 保存到输出 JSON 文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(ordered_data, outfile, indent=4, ensure_ascii=False)

def process_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    for item in data:
        if 'gold_label' in item:
            del item['gold_label']
            del item["shuffled_indices"]
            del item["question_2_id"]
            del item["question_2"]
        item['answer'] = ""  

    if len(data) >= 200:

        sampled_data = random.sample(data, 200)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)


def get_response(
    client: OpenAI,
    content: str,
    model: str ,
    # model: str = args.model_name,
    temperature: float = 1,
):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    response = completion.choices[0].message.content
    assert isinstance(response, str)
    return response

def translate_mixed_question(input_path, output_path):
    client = OpenAI(
    base_url='http://0.0.0.0:23333/v1',

    # required but ignored
    # api_key='ollama',
    )
    stories = open_json(input_path)
    logger.info(f"Load stories from: {input_path}")
    logger.info(f"translate content in: {output_path}")

    data_list = []
    with open(output_path, "a") as f:
        for item in tqdm(stories, desc="translating content"):
            prompt = TRANSLATE_PROMPT.format(
                content1=item["edited_context"],
                content2=item["question_1"],
                content3=item["options"][0],
                content4=item["options"][1],
                content5=item["options"][2],
                content6=item["options"][3],

            )
            del item["answer"]
            model_name = client.models.list().data[0].id
            response = get_response(client, prompt, model_name)
            translated_edited_context = re.search(r"answer1:\s*(.*)", response)
            item["translated_edited_context"] = translated_edited_context.group(1).strip() if translated_edited_context else ""          
            translated_question_1 = re.search(r"answer2:\s*(.*)", response)
            item["translated_question_1"] = translated_question_1.group(1).strip() if translated_question_1 else ""
            translated_options_0 = re.search(r"answer3:\s*(.*)", response)
            item["translated_options_0"] = translated_options_0.group(1).strip() if translated_options_0 else ""
            translated_options_1 = re.search(r"answer4:\s*(.*)", response)
            item["translated_options_1"] = translated_options_1.group(1).strip() if translated_options_1 else ""
            translated_options_2 = re.search(r"answer5:\s*(.*)", response)
            item["translated_options_2"] = translated_options_2.group(1).strip() if translated_options_2 else ""
            translated_options_3 = re.search(r"answer6:\s*(.*)", response)
            item["translated_options_3"] = translated_options_3.group(1).strip() if translated_options_3 else ""

            # item["response"] = response
            item["answer"] = ""

            data_list.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def process(input_path, output_path):
    import chardet 

    with open(input_path, "rb") as f:
        raw_data = f.read()
        encoding = chardet.detect(raw_data)["encoding"]

    # 重新以正确编码读取并转换为 UTF-8
    with open(input_path, "r", encoding=encoding) as f:
        data = f.read()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data)




def calculate_accuracy_human(main_file: str, gold_file: str, output_file: str) -> float:
    """
    计算 JSON 数据中的准确率：
    - 若 answer 由多个值组成（逗号分隔），只要 gold_label 存在于其中，则认为正确。

    参数:
    main_file: str - 包含 mixed_question_id 和 answer 的 JSON 文件路径。
    gold_file: str - 包含 mixed_question_id 和 gold_label 的 JSON 文件路径。

    返回:
    float - 计算得到的准确率（0-1）。
    """
    
    # 读取 JSON 文件
    with open(main_file, "r", encoding="utf-8") as f:
        main_data = json.load(f)
    
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # 创建 gold_label 查找字典 {mixed_question_id: gold_label}
    gold_dict = {item["mixed_question_id"]: str(item["gold_label"]) for item in gold_data}

    # 计算准确率
    correct_count = 0
    total_count = len(main_data)

    # 记录 mixed_question_id、answer 和 gold_answer
    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mixed_question_id", "answer", "gold_answer", "is_correct"])  # 表头
        
        for item in main_data:
            mixed_id = item["mixed_question_id"]
            answer = str(item["answer"]) 
            gold_answer = gold_dict.get(mixed_id, "")  # 若无匹配项，默认空字符串
            
            # 处理多选答案
            answer_set = set(answer.split(","))  # 拆分 answer 为集合
            is_correct = gold_answer in answer_set  # 只要 gold_answer 存在于 answer 中，就算正确

            # 记录到 CSV 文件
            writer.writerow([mixed_id, answer, gold_answer, is_correct])

            if is_correct:
                correct_count += 1

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

def count_question(input_file):
        # 读取 JSON 文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 提取所有 question_1_id
    question_1_ids = [entry["question_1"] for entry in data]

    # 统计不同 question_1_id 的数量
    question_1_count = Counter(question_1_ids)

    # 输出统计结果
    print(len(question_1_count))

def count_words(input_file):
    # 读取 JSON 文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 定义一个函数来统计单词数
    def count_words_in_text(text):
        # 使用 split() 方法将文本拆分成单词，默认按空格拆分
        return len(text.split())

    sum_edited_context_word_count = 0
    sum_question_1_word_count = 0
    sum_options_avg_word_count = 0
    # 统计每个字段的单词数
    for entry in data:
        # 统计每个字段的单词数
        edited_context_word_count = count_words_in_text(entry["edited_context"])
        sum_edited_context_word_count += edited_context_word_count
        question_1_word_count = count_words_in_text(entry["question_1"])
        sum_question_1_word_count += question_1_word_count

        # 分别统计 options 中每个选项的单词数
        options_word_counts = [count_words_in_text(option) for option in entry["options"]]

        # 计算平均单词数
        options_avg_word_count = sum(options_word_counts) / len(options_word_counts) if options_word_counts else 0
        sum_options_avg_word_count += options_avg_word_count

    avg_context = sum_edited_context_word_count / 958
    avg_qestion = sum_question_1_word_count / 958
    avg_option = sum_options_avg_word_count / 958
    print(f"Average Edited Context Word Count: {avg_context:.2f}")
    print(f"Average Question 1 Word Count: {avg_qestion:.2f}")
    print(f"Average Options Word Count: {avg_option:.2f}")
     

def del_explaination_item(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    data_list = []
    mixed_question_num = 0
    for item in data:
        del item["question_contradiction"]

        data_list.append(item)
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=4)

def split_json_file(input_path, output_prefix='common_true_item_part', num_parts=3):
    """
    将一个 JSON 文件拆分为多个部分，并保存为多个 JSON 文件。
    
    参数:
        input_path (str): 原始 JSON 文件路径。
        output_prefix (str): 输出文件名前缀（不包含路径和后缀）。
        num_parts (int): 拆分的份数，默认是3。
    
    返回:
        List[str]: 拆分后的文件路径列表。
    """
    # 读取原始 JSON 数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("输入的 JSON 文件内容必须是一个列表。")
    
    # 计算每份大小
    chunk_size = len(data) // num_parts
    output_files = []
    base_dir = os.path.dirname(input_path)

    # 拆分数据并保存
    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = None if i == num_parts - 1 else (i + 1) * chunk_size
        part_data = data[start_idx:end_idx]
        output_path = os.path.join(base_dir, f"{output_prefix}{i+1}.json")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(part_data, out_f, ensure_ascii=False, indent=2)
        output_files.append(output_path)

    return output_files

def ensure_single_space_between_sentences(text):
    # 识别两个句子之间缺少或多余空格的情况，并统一替换为一个空格
    return re.sub(r'([.!?])(?=[A-Z])', r'\1 ', text)



def calcuate_gold_label_distribution():
    # 读取原始 JSON 文件
    with open('E:\\aaastudy\\A_DIG LLM组\\大模型显隐式混合推理\\数据集构造\\after_option_optimality_mixed_question_final_shuffled_options_gpt4o.json', 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取整个 JSON 数组

    zero_count=0
    one_count=0
    two_count=0
    three_count=0
    # 遍历 JSON 数组中的每个字典
    for item in data:
        if not item:  # 跳过空字典
            continue

        # 获取选项列表和原始最佳选项
        options = item['options']
        original_gold_label = item['gold_label']
        if item['shuffled_indices'][int(item['answer'])]==0:
            zero_count+=1
        if item['shuffled_indices'][int(item['answer'])]==1:
            one_count+=1
        if item['shuffled_indices'][int(item['answer'])]==2:
            two_count+=1
        if item['shuffled_indices'][int(item['answer'])]==3:
            three_count+=1
    print(f'0:{zero_count}, 1:{one_count}, 2:{two_count}, 3:{three_count}')


import json
import random

def shuffle_option(input_file, output_file):
    field_order = [
        "mixed_question_id",
        "roc_passage_id",
        "roc_passage",
        "edited_context",
        "question_1_id",
        "question_1",
        "question_2_id",
        "question_2",
        "options",
        "gold_label",
        "shuffled_indices"
    ]

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for idx, item in enumerate(data):
        if not item:
            continue

        # 更新 mixed_question_id 从 958 开始
        item["mixed_question_id"] = 958 + idx

        # 处理 options 句间空格
        item["options"] = [ensure_single_space_between_sentences(option) for option in item["options"]]
        options = item['options']
        original_gold_label = item['gold_label']

        # 打乱顺序并更新
        shuffled_indices = random.sample(range(len(options)), len(options))
        shuffled_options = [options[i] for i in shuffled_indices]
        new_gold_label = shuffled_indices.index(original_gold_label)

        item["edited_context"] = item["edited_roc_passage_with_q2"]
        item["options"] = shuffled_options
        item["gold_label"] = new_gold_label
        item["shuffled_indices"] = shuffled_indices

        # 删除不需要字段
        for key in [
            "edited_roc_passage_with_q2",
            "question_contradiction", "question_contradiction_explanation",
            "option_contradiction_1", "option_contradiction_explanation_1",
            "option_contradiction_2", "option_contradiction_explanation_2",
            "option_contradiction_3", "option_contradiction_explanation_3",
            "option_contradiction_4", "option_contradiction_explanation_4",
            "success", "issues"
        ]:
            item.pop(key, None)

        # 重新排序字段
        reordered_item = {k: item.get(k) for k in field_order}
        processed_data.append(reordered_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 处理完成，结果已写入 '{output_file}' 文件。")

def convert_jsonl_to_json(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace('.jsonl', '.json')
            output_path = os.path.join(output_folder, output_filename)

            json_array = []
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        try:
                            json_array.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error decoding line in {filename}: {e}")
                            continue

            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(json_array, outfile, ensure_ascii=False, indent=2)

    print("Batch conversion complete.")


import os
import json

def convert_jsonl_to_json_folder(folder_paths):
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Skipped: {folder_path} is not a folder.")
            continue

        print(f"Processing folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".jsonl"):
                jsonl_path = os.path.join(folder_path, filename)
                json_path = os.path.join(folder_path, filename.replace(".jsonl", ".json"))

                data = []
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"  [ERROR] {filename} line {i}: {e}")
                            print(f"  [LINE] {line}")
                            continue  # 忽略该行，继续下一行

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"  Converted: {filename} -> {os.path.basename(json_path)}")

def get_most_frequent_category(type_data):
    if not type_data:
        return None
    return max(type_data.items(), key=lambda x: x[1])[0]

def add_question_type(input_file, reasoning_type_file, output_file):
    with open(reasoning_type_file, 'r', encoding='utf-8') as f:
        reasoning_types = json.load(f)
    with open(input_file, 'r') as f:
        items = json.load(f)
    data_list = []
    for item in items:
        q1_id = item.get("question_1_id")
        q2_id = item.get("question_2_id")
        
        q1_type_data = reasoning_types.get("question_1_reasoning_types", {}).get(q1_id, {})
        q2_type_data = reasoning_types.get("question_2_reasoning_types", {}).get(q2_id, {})
        
        item["question_1_reasoning_type"] = get_most_frequent_category(q1_type_data)
        item["question_2_reasoning_type"] = get_most_frequent_category(q2_type_data)



if __name__ == "__main__":
    # print(post_process_story_ending(text))
    # input_file = "/data/code/MoPS/assets/PS_test/mixed_question_gpt4o.json" #  {'0': 639, '1': 347, '2': 161, '3': 225}
    # input_file = "/data/code/MoPS/assets/PS_test/mixed_question_qwen2.5_72B.json" # {'0': 529, '1': 568, '2': 203, '3': 72}
    # input_file = "/data/code/MoPS/assets/PS_test/mixed_question_llama3.1_70B.json" # {'0': 688, '1': 461, '2': 158, '3': 65}
    # input_file = "/data/code/MoPS/assets/PS_test/mixed_question_qwen2.5_14B.json" # {'0': 719, '1': 436, '2': 151, '3': 65}
    # input_file = "/data/code/MoPS/assets/PS_test/mixed_question_qwen2.5_7B.json" # {'0': 622, '1': 414, '2': 197, '3': 139}
    # input_file = "/data/code/MoPS/assets/PS_test/mixed_question_llama3.1_8B.json" # {'0': 374, '1': 402, '2': 302, '3': 294}
    
    # input_file = "/data/code/MoPS/assets/PS_test/revised_mixed_question_gpt4o.json" # {'0': 579, '1': 233, '2': 121, '3': 131}
    # distribution = count_answer_distribution(input_file)
    # print("Selected content distribution:", distribution)


    # 根据roc_passage_id,提取每个 question_id 对应的 question 和 inverse_option
    # input_file = "/data/code/MoPS/assets/PS_test/edit_story_with_inverse_option_gpt4o.json"
    # output_file ="/data/code/MoPS/assets/PS_test/dict_edit_story_with_inverse_option_gpt4o.json"

    # input_file = "/data/code/MoPS/assets/training_dataset/train_with_inverse_option_gpt4o.json"
    # output_file ="/data/code/MoPS/assets/training_dataset/dict_train_with_inverse_option_gpt4o.json"

    # collect_question(input_file, output_file)

    # 生成混合问题
    # input_file = "/data/code/MoPS/assets/PS_test/dict_edit_story_with_inverse_option_gpt4o.json"
    # output_file ="/data/code/MoPS/assets/PS_test/mixed_question_gpt4o.json"

    # input_file = "/data/code/MoPS/assets/training_dataset/dict_train_with_inverse_option_gpt4o.json"
    # output_file = "/data/code/MoPS/assets/training_dataset/train_mixed_question_gpt4o.json"
    # process_json(input_file, output_file)

    # jsonl_to_json("/data/code/MoPS/assets/PS_test/contradiction_mixed_question_gpt4o.jsonl","/data/code/MoPS/assets/PS_test/contradiction_mixed_question_gpt4o.json")
    # jsonl_to_json("/data/code/MoPS/assets/training_dataset/contradiction_train_mixed_question_gpt4o.jsonl","/data/code/MoPS/assets/training_dataset/contradiction_train_mixed_question_gpt4o.json")
    # del_contradict_item("/data/code/MoPS/assets/PS_test/contradiction_mixed_question_gpt4o_revised.json","/data/code/MoPS/assets/PS_test/revised_mixed_question.json")

    # counts = count_contradiction("/data/code/MoPS/assets/PS_test/revised_mixed_question.json")
    # print("counts:",counts)

    # input_file_path = "/data/code/MoPS/assets/PS_test/edited_mixed_question.json"
    # output_file_path = "/data/code/MoPS/assets/PS_test/edited_mixed_question_copy.json"

    # # 指定键的顺序
    # key_order = [
    #     "mixed_question_id",
    #     "roc_passage_id",
    #     "roc_passage",
    #     "edited_roc_passage_with_q2",
    #     "edited_context",
    #     "question_1_id",
    #     "question_1",
    #     "question_2_id",
    #     "question_2",
    # ]

    # # 调用函数
    # reorder_json_file(input_file_path, output_file_path, key_order)

    # print(f"重新排列后的 JSON 已保存到文件：{output_file_path}")

    
    # input_file_path = "/data/code/MoPS/assets/PS_test/option_optimality_mixed_question_gpt4o_revised.json"
    # count_option_optimality(input_file_path)

    # del_contradict_item("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question.json",
    # "/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final.json")


    # jsonl_to_json("/data/code/MoPS/assets/implicit_setting_new/llama3.1-8b-instruct_new.jsonl",
    # "/data/code/MoPS/assets/implicit_setting_new/llama3.1-8b-instruct_new.json")


    # process_dataset(
    #     "/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options.json",
    #     "/data/code/MoPS/assets/PS_test/mixed_question.json"

    # )
    # translate_mixed_question(
    #     Path("/data/code/MoPS/assets/PS_test/mixed_question.json"),
    #     Path("/data/code/MoPS/assets/PS_test/translated_mixed_question.json")
    # )
    

    # accuracy = calculate_accuracy_human(
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/human-baseline_wl.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/human-baseline_wl_100.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/human-baseline_ymz.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/translated_mixed_question_shq.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/translated_mixed_question_xzc.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/translated_mixed_question_lxx.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/translated_mixed_question_lxx_50.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/translated_mixed_question_wzy.json",
    #     "/data/code/MoPS/assets/explicit_implicit_setting_new/mix_question_100_zsy.json",
    #     "/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options.json",
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/results_wl.csv" # 60
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/results_shq.csv" # 67
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/results_ymz.csv"# 75
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/results_xzc.csv" # 49
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/results_lxx.csv" # 45
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/results_lxx_50.csv" # 46
    #     # "/data/code/MoPS/assets/explicit_implicit_setting_new/results_wzy.csv" # 57.00%
    #     "/data/code/MoPS/assets/explicit_implicit_setting_new/results_zsy.csv" 

    # )
    # print(f"准确率: {accuracy:.2%}")


    # # count_question("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options.json")
    # count_words("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options.json")

    # json_path = "/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_del.json"
    # del_explaination_item(json_path, json_path)


    # jsonl_to_json(
    #     "/data/code/MoPS/assets/training_dataset/train.jsonl",
    #     "/data/code/MoPS/assets/training_dataset/train.json"
    # )
    # jsonl_to_json("/data/code/MoPS/assets/back_forward_thinking_with_reasoning.jsonl","/data/code/MoPS/assets/back_forward_thinking_with_reasoning.json")
    # jsonl_to_json("/data/code/MoPS/assets/implicit_info_verify/deepseek-v3.jsonl","/data/code/MoPS/assets/implicit_info_verify/deepseek-v3.json")
    # jsonl_to_json("/data/code/MoPS/assets/implicit_info_verify/common_true_item.jsonl","/data/code/MoPS/assets/implicit_info_verify/common_true_item.json")
    # jsonl_to_json("/data/code/MoPS/assets/implicit_info_verify/deepseek_merged_output.jsonl","/data/code/MoPS/assets/implicit_info_verify/deepseek_merged_output.json")
    # jsonl_to_json("/data/code/MoPS/assets/implicit_info_verify/qwen_max_merged_output.jsonl","/data/code/MoPS/assets/implicit_info_verify/qwen_max_merged_output.json")
    # jsonl_to_json("/data/code/MoPS/assets/implicit_info_verify/common_item.jsonl","/data/code/MoPS/assets/implicit_info_verify/common_item.json")

    # split_json_file('/data/code/MoPS/assets/implicit_info_verify/common_true_item.json')

    # text = "Fred felt bad that he won, so the next game he eased up and let James win.James learned that day to focus on the game, not on bragging."
    # fixed_text = ensure_single_space_between_sentences(text)
    # print(fixed_text)

    # input_file = "/data/code/MoPS/assets/implicit_info_verify/common_item.json"
    # output_file = "/data/code/MoPS/assets/implicit_info_verify/mixed_question_final_shuffled_options.json"
    # shuffle_option(input_file, output_file)

    # convert_jsonl_to_json(
    #     "/data/code/MoPS/assets/double_explicit_setting_2246",
    #     "/data/code/MoPS/assets/double_explicit_setting_2246_json"
    # )

    # convert_jsonl_to_json(
    #     "/data/code/MoPS/assets/question_classification",
    #     "/data/code/MoPS/assets/double_explicit_setting_2246_json"
    # )
    # jsonl_to_json(
    #     "/data/code/MoPS/assets/question_classification/qwen-max-2025-01-25_2-16.jsonl",
    #     "/data/code/MoPS/assets/question_classification/qwen-max-2025-01-25_2-16.json"
    # )

    # jsonl_to_json(
    #     "/data/code/MoPS/assets/single_sentence_question/llama3_1-70b-instruct-with-new-questions.jsonl",
    #     "/data/code/MoPS/assets/single_sentence_question/llama3_1-70b-instruct-with-new-questions.json"
    # )

    # folders = [
    # # "/data/code/MoPS/assets/double_explicit_setting_2246",
    # # "/data/code/MoPS/assets/explicit_implicit_setting_2246",
    # # "/data/code/MoPS/assets/explicit_implicit_setting_one_shot_2246",
    # # "/data/code/MoPS/assets/explicit_setting_2246",
    # # "/data/code/MoPS/assets/implicit_setting_2246",
    # # "/data/code/MoPS/assets/explicit_implicit_setting_one_shot_2246",
    # # "/data/code/MoPS/assets/inference_between_options_and_sentences"
    # # "/data/code/MoPS/assets/explicit_implicit_setting_cot_2246",
    # # "/data/code/MoPS/assets/select_content_with_options_related_sentences",
    # # "/data/code/MoPS/assets/single_sentence_question"
    # # "/data/code/MoPS/assets/single_sentence_question"
    # # "/data/code/MoPS/assets/reasoning_sentences"
    # # "/data/code/MoPS/assets/select_content_with_new_questions"
    # # "/data/code/MoPS/assets/select_content_with_reasoning_sentences"
    # "/data/code/MoPS/assets/explicit_implicit_setting_2246"

    # ]

    # convert_jsonl_to_json_folder(folders)


    add_question_type(
        "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json",
        "/data/code/MoPS/assets/question_classification/merged-reasoning_type_distribution.json",
        "/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-with-reasoning-type.json"
    )