# MixRea

You need to add your api-key in .env to run our codes.

Alibaba Cloud API and Openai API can be invoked through `multiprocess.py` and `multiprocess_gpt.py`. 

Functions for specific inference tasks can be imported from `induce.py`.

`MixRea.json` is our dataset. The attributes are described below.

|     Field Name      |  Type   |                         Description                          |
| :-----------------: | :-----: | :----------------------------------------------------------: |
| `mixed_question_id` | integer |           Unique identifier for the mixed question           |
|  `roc_passage_id`   | string  |   Unique identifier for the original passage (UUID format)   |
|    `roc_passage`    | string  | Original passage text describing a traffic incident involving a rock |
|  `edited_context`   | string  | Enhanced context with additional details about the driver changing a tire |
|   `question_1_id`   | string  |           Unique identifier for the first question           |
|    `question_1`     | string  |                    The explicit question                     |
|   `question_2_id`   | string  |          Unique identifier for the second question           |
|    `question_2`     | string  |          The implicit question for context revision          |
|      `options`      |  array  | Four possible answer choices describing different outcome scenarios |
|    `gold_label`     | integer | Index of the correct answer (0-based) in the original `options` array |
| `shuffled_indices`  |  array  | Randomized display order of options, indicating their initial sequence |

`Prompts.py` contains all the prompt templates used for dataset construction and experimental testing in the project. This module serves as the central repository for prompt engineering across different tasks and model interactions.

`result/` directory systematically organizes all model response outputs from experiments, with subfolders categorized by specific inference task configurations.
