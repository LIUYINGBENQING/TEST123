# MixRea

## `MixRea.json` is our dataset. The attributes are described below.

|     Field Name      |  Type   |                         Description                          |
| :-----------------: | :-----: | :----------------------------------------------------------: |
| `mixed_question_id` | integer |           Unique identifier for the mixed question           |
|  `roc_passage_id`   | string  |          Unique identifier for the initial context           |
|    `roc_passage`    | string  |                       Initial context                        |
|  `edited_context`   | string  |            Context with the implicit information             |
|   `question_1_id`   | string  | Unique identifier for the question with explicit information |
|    `question_1`     | string  |              Question with explicit information              |
|   `question_2_id`   | string  | Unique identifier for the question with implicit information |
|    `question_2`     | string  |              Question with implicit information              |
|      `options`      |  array  |    Four options describing different possible story plots    |
|    `gold_label`     | integer | Index of the correct answer in the original `options` array [0,1,2,3] |
| `shuffled_indices`  |  array  | Randomized display order of options, indicating their initial sequence |

## Experimental Results

`Experimental Results` directory systematically organizes all model response outputs from experiments, with subfolders categorized by specific inference task configurations. The  directory tree are shown below.

```
-Experimental Results
	-explicit_implicit_setting_2246 (is used in Figure 5, Table 2, and Figure 7)
	-duel_explicit_setting_2246 (is used in Figure 7)
	-explicit_setting_2246 (is used in Figure 7)
	-implicit_setting_2246 (is used in Figure 7)
	-explicit_implicit_setting_cot_2246  (is used in Table 3)
	-explicit_implicit_setting_one_shot_2246   (is used in Table 3)
	-explicit_implicit_setting_prcp_2246   (is used in Table 3)
		-step_1_question_generation
		-step_2_reasoning_sentences
		-step_3_prcp_without_question_selection
		-step_3_prcp_with_question_selection
```

## Prompts

`Prompts.py` contains all the prompt templates used for dataset construction and experimental testing in the project.

## Codes

`Code.py` contains all the codes used in the project.
