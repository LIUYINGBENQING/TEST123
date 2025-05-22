import multiprocessing
from pathlib import Path
from openai import OpenAI
import os
from mops.induce import select_content_with_implicit_explicit_setting, dashscope_select_content_with_implicit_explicit_setting
# from mops.constants import client 
# client = OpenAI(
#     api_key=os.getenv('ALIYUN_API_KEY'), 
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="xxx",
)

# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_qwen2-72b-instruct.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_qwen-max-2025-01-25_2-16.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_qwen2-7b-instruct.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_qwen2.5-7b-instruct.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_qwen2.5-32b-instruct.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-gemma3-12b.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-llama3_1_405b.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-gemma3-27b.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-llama3-70b.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_llama3_1_70b.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_claude.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_r1.json")
input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246_gemini_pro.json")
params_list = [

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/deepseek-r1.json","model_name":"deepseek-r1"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/deepseek-v3.json","model_name":"deepseek-v3"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen-max-2025-01-25_2-16.json","model_name":"qwen-max-2025-01-25"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-72b-instruct_2-16_2.json","model_name":"qwen2.5-72b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-32b-instruct_2-16.json","model_name":"qwen2.5-32b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-14b-instruct_2-16.json","model_name":"qwen2.5-14b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-7b-instruct_2-16_2.json","model_name":"qwen2.5-7b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-3b-instruct_2-16.json","model_name":"qwen2.5-3b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-1.5b-instruct_2-16.json","model_name":"qwen2.5-1.5b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2-72b-instruct.json","model_name":"qwen2-72b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2-7b-instruct.json","model_name":"qwen2-7b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/llama3.1-405b-instruct.json","model_name":"llama3.1-405b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/llama3.1-70b-instruct.json","model_name":"llama3.1-70b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/llama3.1-8b-instruct_2-16.json","model_name":"llama3.1-8b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/llama3-70b-instruct.json","model_name":"llama3-70b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/llama3-8b-instruct.json","model_name":"llama3-8b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/llama2-13b-chat-v2.json","model_name":"llama2-13b-chat-v2"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/llama2-7b-chat-v2.json","model_name":"llama2-7b-chat-v2"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/baichuan2-13b-chat-v1.json","model_name":"baichuan2-13b-chat-v1"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/baichuan2-7b-chat-v1.json","model_name":"baichuan2-7b-chat-v1"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_additional/gemma-3-1b.json","model_name":"google/gemma-3-1b-it:free"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_additional/gemma-3-4b.json","model_name":"google/gemma-3-4b-it"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_additional/gemma-2-9b.json","model_name":"google/gemma-2-9b-it"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_additional/glm-4-32b.json","model_name":"thudm/glm-4-32b"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_additional/glm-4-9b.json","model_name":"thudm/glm-4-9b:free"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/gemma-2-27b.json","model_name":"google/gemma-2-27b-it"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/gemma-3-12b.jsonl","model_name":"google/gemma-3-12b-it"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/gemma-3-27b.jsonl","model_name":"google/gemma-3-27b-it"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/llama3_1-70b-instruct.jsonl","model_name":"meta-llama/llama-3.1-70b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/llama3_1-8b-instruct.json","model_name":"meta-llama/llama-3.1-8b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/llama3_1-405b-instruct.jsonl","model_name":"meta-llama/llama-3.1-405b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/llama3-70b-instruct.jsonl","model_name":"meta-llama/llama-3-70b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/llama3-8b-instruct.json","model_name":"meta-llama/llama-3-8b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/qwen-max-2025-01-25_2-16.jsonl","model_name":"qwen-max-2025-01-25"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/qwen2.5-72b-instruct.json","model_name":"qwen2.5-72b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/qwen2.5-32b-instruct.jsonl","model_name":"qwen2.5-32b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/qwen2.5-7b-instruct.jsonl","model_name":"qwen2.5-7b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/qwen2-72b-instruct.jsonl","model_name":"qwen2-72b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/qwen2-7b-instruct.jsonl","model_name":"qwen2-7b-instruct"},

    {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/gemini-2.5-pro-preview.jsonl","model_name":"google/gemini-2.5-pro-preview"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/gemini-2.5-flash-preview.jsonl","model_name":"google/gemini-2.5-flash-preview"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/claude-3.7-sonnet.jsonl","model_name":"anthropic/claude-3.7-sonnet"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/claude-3.7-sonnet_thinking.jsonl","model_name":"anthropic/claude-3.7-sonnet:thinking"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/deepseek-r1.jsonl","model_name":"deepseek/deepseek-r1"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/deepseek-chat-v3-0324.jsonl","model_name":"deepseek/deepseek-chat-v3-0324"},


]

def run_action(params):
    try:

        # dashscope_select_content_with_implicit_explicit_setting(input_path, params["output_path"], client, params["model_name"])
        select_content_with_implicit_explicit_setting(input_path, params["output_path"], client, params["model_name"])
        return f"Success: {params['model_name']}"
    except Exception as e:
        return f"Error in {params['model_name']}: {str(e)}"


if __name__ == "__main__":
    with multiprocessing.Pool(processes=len(params_list)) as pool:
        results = pool.map(run_action, params_list)

    for result in results:
        print(result)