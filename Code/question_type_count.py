import json
from collections import defaultdict

def save_reasoning_type_distribution(data, output_filename="reasoning_type_distribution.json"):
    """
    统计并保存每个问题的推理类型分布到 JSON 文件。

    参数:
    - data: 包含多个字典的列表，每个字典代表一个问题的 JSON 数据。
    - output_filename: 结果保存的文件名，默认保存为 'reasoning_type_distribution.json'。
    """
    # 使用 defaultdict 来存储统计结果
    question1_counter = defaultdict(lambda: defaultdict(int))  # 以 question_1_id 为键，推理类型为值
    question2_counter = defaultdict(lambda: defaultdict(int))  # 以 question_2_id 为键，推理类型为值

    # 遍历数据并统计每个问题的推理类型
    for item in data:
        # print(item)
        question1_counter[item['question_1_id']][item['question1_reasoning_type']] += 1
        question2_counter[item['question_2_id']][item['question2_reasoning_type']] += 1

    # 将结果保存到 JSON 文件中
    output_data = {
        "question_1_reasoning_types": dict(question1_counter),
        "question_2_reasoning_types": dict(question2_counter)
    }

    # 保存到文件
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=2)

    print(f"统计结果已保存到 {output_filename}")

def merge_json_files(file_list, output_file):
    # Initialize an empty dictionary to store the merged data
    merged_data = {
        "question_1_reasoning_types": defaultdict(lambda: defaultdict(int)),
        "question_2_reasoning_types": defaultdict(lambda: defaultdict(int)),
    }

    # Iterate through each file and merge the data
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Merge the "question_1_reasoning_types" data
            for question_id, reasoning_types in data.get("question_1_reasoning_types", {}).items():
                for reasoning_type, count in reasoning_types.items():
                    merged_data["question_1_reasoning_types"][question_id][reasoning_type] += count

            # Merge the "question_2_reasoning_types" data
            for question_id, reasoning_types in data.get("question_2_reasoning_types", {}).items():
                for reasoning_type, count in reasoning_types.items():
                    merged_data["question_2_reasoning_types"][question_id][reasoning_type] += count

    # Write the merged data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merged data has been saved to {output_file}")




if __name__ == "__main__":

    # with open("/data/code/MoPS/assets/question_classification/qwen-max-2025-01-25_2-16.json", "r") as f:
    #     data = json.load(f)
# 
    # save_reasoning_type_distribution(data, "/data/code/MoPS/assets/question_classification/deepseek-reasoning_type_distribution.json")
    # save_reasoning_type_distribution(data, "/data/code/MoPS/assets/question_classification/gpt-4o_type_distribution.json")
    # save_reasoning_type_distribution(data, "/data/code/MoPS/assets/question_classification/qwen-max-reasoning_type_distribution.json")
    # add_to_dict("/data/code/MoPS/assets/question_classification/deepseek-reasoning_type_distribution.json", "/data/code/MoPS/assets/question_classification/gpt-4o-2024-08-06_2-16.json", "/data/code/MoPS/assets/question_classification/reasoning_type_distribution_with_all.json")


    file_list = [
        '/data/code/MoPS/assets/question_classification/deepseek-reasoning_type_distribution.json',
        '/data/code/MoPS/assets/question_classification/gpt-4o_type_distribution.json',
        '/data/code/MoPS/assets/question_classification/qwen-max-reasoning_type_distribution.json'
    ]
    output_file = '/data/code/MoPS/assets/question_classification/merged-reasoning_type_distribution.json'

    merge_json_files(file_list, output_file)