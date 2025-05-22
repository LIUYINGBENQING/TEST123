import multiprocessing
from pathlib import Path
from mops.induce import select_content_with_double_explicit_setting
from openai import OpenAI
import os
# from mops.constants import client 

# client = OpenAI(
#     api_key="xxx",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key="xxx",
# )
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://api.openai.com/v1",
)

# input_path = Path("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options.json")
input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_double_explicit.json")

params_list = [
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/deepseek-r1.json","model_name":"deepseek-r1"},

    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/deepseek-v3.json","model_name":"deepseek-v3"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/qwen-max-2025-01-25.json","model_name":"qwen-max-2025-01-25"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/qwen2.5-72b-instruct.json","model_name":"qwen2.5-72b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/qwen2.5-7b-instruct.json","model_name":"qwen2.5-7b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/llama3.1-405b-instruct.json","model_name":"llama3.1-405b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/llama3.1-70b-instruct.json","model_name":"llama3.1-70b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_new/llama3.1-8b-instruct_new.json","model_name":"llama3.1-8b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/llama-3.1-8b-instruct.jsonl","model_name":"meta-llama/llama-3.1-8b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/llama-3.1-70b-instruct.jsonl","model_name":"meta-llama/llama-3.1-70b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/llama-3.1-405b-instruct.jsonl","model_name":"meta-llama/llama-3.1-405b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/deepseek-chat-v3-0324.jsonl","model_name":"deepseek/deepseek-chat-v3-0324"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/gemini-2.5-flash-preview.jsonl","model_name":"google/gemini-2.5-flash-preview"},

    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/qwen-max-2025-01-25_2-16.jsonl","model_name":"qwen-max-2025-01-25"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/qwen2.5-72b-instruct.jsonl","model_name":"qwen2.5-72b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/qwen2.5-7b-instruct.jsonl","model_name":"qwen2.5-7b-instruct"},

    {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/gpt-4o-2024-08-06_2-16.jsonl","model_name":"gpt-4o-2024-08-06"},
    {"output_path":"/data/code/MoPS/assets/double_explicit_setting_2246/gpt-4o-mini-2024-07-18.jsonl","model_name":"gpt-4o-mini-2024-07-18"},

]

def run_action(params):
    try:
        select_content_with_double_explicit_setting(input_path, params["output_path"], client, params["model_name"])
        return f"Success: {params['model_name']}"
    except Exception as e:
        return f"Error in {params['model_name']}: {str(e)}"


if __name__ == "__main__":
    with multiprocessing.Pool(processes=len(params_list)) as pool:
        results = pool.map(run_action, params_list)
        
    for result in results:
        print(result)