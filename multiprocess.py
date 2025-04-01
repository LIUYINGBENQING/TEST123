import multiprocessing
from pathlib import Path
from openai import OpenAI
from mops.induce import select_content_with_implicit_explicit_setting, dashscope_select_content_with_implicit_explicit_setting
# from mops.constants import client 
client = OpenAI(
    api_key=os.getenv('ALIYUN_API_KEY'), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
input_path = Path("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options.json")
params_list = [

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/deepseek-r1.json","model_name":"deepseek-r1"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/deepseek-v3.json","model_name":"deepseek-v3"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen-max-2025-01-25_2-16.json","model_name":"qwen-max-2025-01-25"},
    {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-72b-instruct_2-16_2.json","model_name":"qwen2.5-72b-instruct"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-32b-instruct_2-16.json","model_name":"qwen2.5-32b-instruct"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-14b-instruct_2-16.json","model_name":"qwen2.5-14b-instruct"},
    {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/qwen2.5-7b-instruct_2-16_2.json","model_name":"qwen2.5-7b-instruct"},
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