import json
from tqdm import tqdm

def compute_accuracy(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)


    correct_predictions = 0
    for entry in tqdm(data, desc="Computing Accuracy"):
        if entry.get("option_optimality") not in ["0", "1", "2", "3"]:
            continue
        if int(entry.get("option_optimality")) == entry.get("gold_label"):
            correct_predictions+=1


    accuracy = correct_predictions / 958
    return accuracy

def count_matching_option_optimality(file1_path, file2_path,output_path):
    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(file2_path, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    if len(data1) != len(data2):
        print("警告：两个 JSON 文件的 item 数量不同！")
        return None
    
    matching_count = 0
    matching_pairs = []
    
    for item1, item2 in zip(data1, data2):
        # 检查 mixed_question_id 是否相同（可选，确保顺序一致）
        if item1.get("mixed_question_id") != item2.get("mixed_question_id"):
            print(f"警告：mixed_question_id 不匹配！{item1.get('mixed_question_id')} != {item2.get('mixed_question_id')}")
            continue
        
        # 比较 option_optimality
        if item1.get("option_optimality") == item2.get("option_optimality"):
            matching_count += 1
            matching_pairs.append({
                "mixed_question_id": item1.get("mixed_question_id"),
                "option_optimality": item1.get("option_optimality"),
                "gold_label": item1.get("gold_label")
            })
    
    with open(output_path, "w") as f:
        json.dump(matching_pairs, f, indent=4)

    return {
        "total_matching_pairs": matching_count,
        # "matching_pairs": matching_pairs  # 可选：列出匹配的项
    }

def compare_three_files_ordered(file1_path, file2_path, file3_path,output_path):

    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(file2_path, 'r', encoding='utf-8') as f2, \
         open(file3_path, 'r', encoding='utf-8') as f3:
        data1 = json.load(f1)
        data2 = json.load(f2)
        data3 = json.load(f3)
    
    if len(data1) != len(data2) or len(data2) != len(data3):
        print("错误：三个 JSON 文件的 item 数量不一致！")
        return None
    
    matching_count = 0
    matching_items = []
    
    # 按索引顺序遍历三个列表
    for idx, (item1, item2, item3) in enumerate(zip(data1, data2, data3)):
        # 检查 mixed_question_id 是否一致（可选）
        if (item1.get("mixed_question_id") != item2.get("mixed_question_id") or 
            item2.get("mixed_question_id") != item3.get("mixed_question_id")):
            print(f"警告：索引 {idx} 处的 mixed_question_id 不一致！")
            continue
        
        # 检查 option_optimality 是否全部相同
        opt1 = item1.get("option_optimality")
        opt2 = item2.get("option_optimality")
        opt3 = item3.get("option_optimality")
        
        if opt1 == opt2 == opt3:
            matching_count += 1
            matching_items.append({
                "mixed_question_id": item1.get("mixed_question_id"),
                "option_optimality": opt1,
                "gold_label": item1.get("gold_label")
            })
    
    with open(output_path, "w") as f:
        json.dump(matching_items, f, indent=4)
    return {
        "total_matching_items": matching_count,
    }

if __name__ == "__main__":

    json_file_path_gpt4o = "/data/code/MoPS/assets/PS_test/option_optimality_gpt4o_3-31.json" # acc:0.7651356993736952
    json_file_path_qwen = "/data/code/MoPS/assets/PS_test/option_optimality_qwen-max-2025-01-25_3-31.json" #acc: 0.7233820459290188
    json_file_path_ds = "/data/code/MoPS/assets/PS_test/option_optimality_deepseek-v3_3-31.json" # acc:0.7599164926931107

    output_path_gpt4o = "/data/code/MoPS/assets/PS_test/option_optimality_gpt4o_3-31_accuracy.json"
    output_path_qwen = "/data/code/MoPS/assets/PS_test/option_optimality_qwen-max-2025-01-25_3-31_accuracy.json"
    output_path_ds = "/data/code/MoPS/assets/PS_test/option_optimality_deepseek-v3_3-31_accuracy.json"
    # acc = compute_accuracy(json_file_path_ds, output_path_ds)
    # print(f"Accuracy: {acc}")

    output_path_gpt4o_qwen = "/data/code/MoPS/assets/PS_test/option_optimality_gpt4o_qwen_matching.json"
    output_path_gpt4o_ds = "/data/code/MoPS/assets/PS_test/option_optimality_gpt4o_ds_matching.json"
    output_path_qwen_ds = "/data/code/MoPS/assets/PS_test/option_optimality_qwen_ds_matching.json"

    # matching_pairs_gpt4o_qwen = count_matching_option_optimality(json_file_path_gpt4o, json_file_path_qwen, output_path_gpt4o_qwen)
    # matching_pairs_gpt4o_ds = count_matching_option_optimality(json_file_path_gpt4o, json_file_path_ds, output_path_gpt4o_ds)
    # matching_pairs_qwen_ds = count_matching_option_optimality(json_file_path_qwen, json_file_path_ds, output_path_qwen_ds)

    acc_gpt4o_qwen = compute_accuracy(output_path_gpt4o_qwen) # gpt4o_qwen acc: 0.6409185803757829
    acc_gpt4o_ds = compute_accuracy(output_path_gpt4o_ds) # gpt4o_ds acc: 0.6711899791231732
    acc_qwen_ds = compute_accuracy(output_path_qwen_ds) # qwen_ds acc: 0.6461377870563675

    print(f"gpt4o_qwen acc: {acc_gpt4o_qwen}")
    print(f"gpt4o_ds acc: {acc_gpt4o_ds}")
    print(f"qwen_ds acc: {acc_qwen_ds}")

    output_path_gpt4o_qwen_ds = "/data/code/MoPS/assets/PS_test/option_optimality_gpt4o_qwen_ds_matching.json"
    # matching_items = compare_three_files_ordered(json_file_path_gpt4o, json_file_path_qwen, json_file_path_ds,output_path_gpt4o_qwen_ds) # 646
    # print(f"matching_items: {matching_items}")
    acc_gpt4o_qwen_ds = compute_accuracy(output_path_gpt4o_qwen_ds)
    print(f"gpt4o_qwen_ds acc: {acc_gpt4o_qwen_ds}") # gpt4o_qwen_ds acc: 0.5908141962421712


    