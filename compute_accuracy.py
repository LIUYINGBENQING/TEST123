import json

def compare_gold_and_answer(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    correct_counter =0
    for item in data:
        try:
            gold = item['gold_label']
            answer = item['answer']
        except KeyError as e:
            print(f"字典缺少必要键: {e}")
            results.append(False)
            continue

        # 尝试将answer转换为整数进行比较
        try:
            answer_int = int(answer)
            is_equal = (gold == answer_int)
        except ValueError:
            # 转换失败则直接比较字符串
            is_equal = (str(gold) == answer)

        if is_equal==True:
            correct_counter+=1
        results.append(is_equal)

    return correct_counter/len(data)


# 示例使用
if __name__ == "__main__":
    results = compare_gold_and_answer('E:/aacodeset/LLM_eval/mops/explicit_implicit_setting_new/deepseek-r1.json')
    print(results)