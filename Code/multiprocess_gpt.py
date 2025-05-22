import multiprocessing
from pathlib import Path
from openai import OpenAI
import os
from mops.induce import select_content_with_implicit_setting, select_content_with_explicit_setting, select_content_with_implicit_explicit_setting



client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://api.openai.com/v1",
)

# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246.json")
# input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-gpt-4o.json")
input_path = Path("/data/code/MoPS/assets/dataset-2246/mixed_question_final_shuffled_options-2246-gpt4o-mini.json")
# input_path = Path("/data/code/MoPS/assets/PS_test/after_option_optimality_mixed_question_final_shuffled_options_ex_im.json")
params_list = [

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/gpt-4o-2024-08-06_2-16.json","model_name":"gpt-4o-2024-08-06"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/o1-2024-12-17.json","model_name":"o1-2024-12-17"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/o3-mini-2025-01-31.json","model_name":"o3-mini-2025-01-31"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/o1-preview-2024-09-12.jsonl","model_name":"o1-preview-2024-09-12"},
    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_new/gpt-4o-mini-2024-07-18.jsonl","model_name":"gpt-4o-mini-2024-07-18"},

    # {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/gpt-4o-2024-08-06_2-16.jsonl","model_name":"gpt-4o-2024-08-06"},
    {"output_path":"/data/code/MoPS/assets/explicit_implicit_setting_2246/gpt-4o-mini-2024-07-18.jsonl","model_name":"gpt-4o-mini-2024-07-18"},
    
]

def run_action(params):
    try:
        select_content_with_implicit_explicit_setting(input_path, params["output_path"], client, params["model_name"])
        return f"Success: {params['model_name']}"
    except Exception as e:
        return f"Error in {params['model_name']}: {str(e)}"


if __name__ == "__main__":
    with multiprocessing.Pool(processes=len(params_list)) as pool:
        results = pool.map(run_action, params_list)

    for result in results:
        print(result)