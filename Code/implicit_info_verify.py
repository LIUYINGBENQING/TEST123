import multiprocessing
from pathlib import Path
from openai import OpenAI
from mops.induce import implicit_info_verify,detect_contradiction
import os
import concurrent.futures
from functools import partial
import json
# from mops.constants import client 
client = OpenAI(
    api_key=os.getenv('ALIYUN_API_KEY'), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key="xxx",
# )
# client = OpenAI(
#     api_key=os.getenv('DEEPSEEK_API_KEY'), 
#     base_url="https://api.deepseek.com/v1",
# )
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url='https://api.openai.com/v1',
# )

# input_path = Path("/data/code/MoPS/assets/training_dataset/contradiction_train_mixed_question_gpt4o_2.json")
# params_list = [

#     {"output_path":"/data/code/MoPS/assets/implicit_info_verify/deepseek-v3.json","model_name":"deepseek-chat"},

#     # {"output_path":"/data/code/MoPS/assets/implicit_info_verify/qwen-max-2025-01-25_2-16.json","model_name":"qwen-max-2025-01-25"},


#     # {"output_path":"/data/code/MoPS/assets/implicit_info_verify/gpt4o.json","model_name":"gpt-4o"},


# ]

# def run_action(params):
#     try:

#         implicit_info_verify(input_path, params["output_path"], client, params["model_name"])
#         return f"Success: {params['model_name']}"
#     except Exception as e:
#         return f"Error in {params['model_name']}: {str(e)}"


def run_action(params):
    input_path, output_path = params
    from mops.induce import implicit_info_verify,detect_contradiction # 确保函数和 client 可以被进程导入
    detect_contradiction(Path(input_path), Path(output_path), client)
    return output_path  # 返回处理后的路径

def batch_detect_contradiction_with_multiprocessing(input_paths, temp_output_paths, merged_output_path):
    assert len(input_paths) == len(temp_output_paths)
    params_list = list(zip(input_paths, temp_output_paths))

    # 多进程并行执行
    with multiprocessing.Pool(processes=len(params_list)) as pool:
        results = pool.map(run_action, params_list)

    # 合并所有 JSONL 文件
    with open(merged_output_path, "w", encoding="utf-8") as merged_f:
        for result_file in results:
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    merged_f.write(line)
    print(f"合并完成，保存至: {merged_output_path}")


if __name__ == "__main__":
    # with multiprocessing.Pool(processes=len(params_list)) as pool:
    #     results = pool.map(run_action, params_list)

    # for result in results:
    #     print(result)

    input_paths = [
        # "/data/code/MoPS/assets/implicit_info_verify/common_true_item_part1_ds.json",
        # "/data/code/MoPS/assets/implicit_info_verify/common_true_item_part2_ds.json",
        # "/data/code/MoPS/assets/implicit_info_verify/common_true_item_part3_ds.json",
        "/data/code/MoPS/assets/implicit_info_verify/common_true_item_part1_qwen.json",
        "/data/code/MoPS/assets/implicit_info_verify/common_true_item_part2_qwen.json",
        "/data/code/MoPS/assets/implicit_info_verify/common_true_item_part3_qwen.json",
    ]

    temp_output_paths = [
        # "/data/code/MoPS/assets/implicit_info_verify/deepseek_v3_output_part1.jsonl",
        # "/data/code/MoPS/assets/implicit_info_verify/deepseek_v3_output_part2.jsonl",
        # "/data/code/MoPS/assets/implicit_info_verify/deepseek_v3_output_part3.jsonl",
        "/data/code/MoPS/assets/implicit_info_verify/qwen_output_part1.jsonl",
        "/data/code/MoPS/assets/implicit_info_verify/qwen_output_part2.jsonl",
        "/data/code/MoPS/assets/implicit_info_verify/qwen_output_part3.jsonl",
    ]

    # merged_output_path = Path("/data/code/MoPS/assets/implicit_info_verify/deepseek_merged_output.jsonl")
    merged_output_path = Path("/data/code/MoPS/assets/implicit_info_verify/qwen_max_merged_output.jsonl")

    batch_detect_contradiction_with_multiprocessing(
        input_paths=input_paths,
        temp_output_paths=temp_output_paths,
        merged_output_path=merged_output_path,
    )