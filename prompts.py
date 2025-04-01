BACKGROUND_PROMPT = """Tell me 10 backgrounds in {theme} themed novels and scripts.

Each background should only include {component} behind literary works and no any other extra narratives. 

Each line starts with a serial number and a dot. 
"""


PROTAGONIST_PROMPT = """The following is the theme and background of a novel or script:

### Theme
{theme}

### Background
{background}

Based on the theme and background mentioned above, tell me 3 possible protagonists.
The protagonist is the main character portrayed in the narratives about their growth.
Each protagonist should only include a brief characterization, without specific names.
"""


PROTAGONIST_ANTAGONIST_PROMPT = """The following is the theme and background of a novel or script:

### Theme
{theme}

### Background
{background}

Based on the theme and background mentioned above, tell me 3 possible (protagonist, antagonist) pairs.
The protagonist is the main character portrayed in the narratives about their growth.
The main role of the antagonist is to create a conflict event with the protagonist to prevent it from achieving its goal.
Each pair should be presented in the format: protagonist: <a brief characterization>; antagonist: <a brief characterization>.
Please remember to use protagonist and antagonist without specific names appearing.
"""


PROTAGONIST_DEUTERAGONIST_PROMPT = """The following is the theme and background of a novel or script:

### Theme
{theme}

### Background
{background}

Based on the theme and background mentioned above, tell me 3 possible (protagonist, deuteragonist) pairs.
The protagonist is the main character portrayed in the narratives about their growth.
The main role of the deuteragonist is to collaborate with the protagonist to achieve its goal.
Each pair should be presented in the format: protagonist: <a brief characterization>; deuteragonist: <a brief characterization>.
Please remember to use protogonist and deuteragonist without specific names appearing.
"""


EVENT_PROMPT = """The following is the theme, background and persona of a novel or script:

### Theme
{theme}

### Background
{background}

### Persona
{persona}

Based on the theme, background and persona mentioned above, conceive two independent events that could run through the entire narrative context.
Please use a concise and coherent sentence to describe the entire event.
"""


ENDING_PROMPT = """The following is the theme, background, persona and main event of a novel or script:

### Theme
{theme}

### Background
{background}

### Persona
{persona}

## Event
{event}

Based on the theme, background, persona and event mentioned above, conceive an concretized ending.
Please use a concise and coherent sentence to describe the ending.
"""


TWIST_PROMPT = """The following is the theme, background, persona, main event and ending of a novel or script:

### Theme
{theme}

### Background
{background}

### Persona
{persona}

## Event
{event}

## Ending
{ending}

Based on the theme, background, persona, event and ending mentioned above, conceive a twist as an unique hook to connect the main event and ending.
Please use a concise and coherent sentence to describe the twist.
"""


SYNYHESIZE_PROMPT = """The following is the theme, background, persona, main event, final ending and twist of a novel or script:

### Theme
{theme}

### Background
{background}

### Persona
{persona}

## Event
{event}

## Ending
{ending}

## Twist
{twist}

Please combine the aforementioned elements of a novel or script into one compact, concise, and coherent sentence as a story premise.
"""


VERIFY_PROMPT = """Here is a story premise:

{premise}

Please help to verify:

1. Does it contain obvious inconsistencies. For example, the background, plot, and characters do not match

2. Does it contain obvious factual errors. For example, there were obvious historical errors and time span errors

If there are any errors mentioned above, please return Yes wrapped by `[[]]`, otherwise return No wrapped by `[[]]` without any other extra output.
"""


FASCINATION_SCORE_PROMPT = """Here is a story premise:

{premise}

Now let's give you a score from 0 to 100 to assess to its fascination.

Score 0 indicates that this premise is completely confused, while score 100 indicates that you really want to see the story created based on this premise.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:"""


ORIGINALITY_SCORE_PROMPT = """Here is a story premise:

{premise}

Now let you give a score from 0 to 100 which represents your level of familiarity with it.

Score 0 indicates that you have seen the exact same premise, while score 100 indicates that you have never seen the same premise at all.

Your score should be based on the assumption that the candidate is at least a complete story premise. Otherwise, you should give a score 0.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:"""


COMPLETENESS_SCORE_PROMPT = """Here is a story premise:

{premise}

Now let's give you a score from 0 to 100 which represents its completeness level.

Score 0 indicates that it lacks all elements , while score 100 indicates that it has all elements.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:"""



### /data/code/Recurrent-LLM/prompts

SYNYHESIZE_STORY_PROMPT = """The following is the theme, background, persona and relationship between persona of a novel or script:

### Theme
{theme}

### Background
{background}

### Persona
{persona}

## Relationship between persona
{relationship}


Please combine the aforementioned elements to generate a fable of about 5 sentences.

Very Important:
Remember that you are writing a fable and convey a moral lesson.
"""













GENERATE_STORY_ENDING_PROMPT = """

The following is the beginning of s story:

### Story Beginning
{story}

Event: Add a brief but meaningful development that moves the story closer to the end.

Twist: Introduce a small, unexpected turn to make the ending more interesting.

Ending: Conclude quickly with a resolution that feels natural and provides closure.

**Instructions**:  
Write a short continuation that introduces a quick development, twist, and ending.

**Output**:  
Only provide the completed, brief story.

"""




EDIT_POSSIBLE_STORY_PROMPT = """
The following is a story, a question about the story, and an ending. 

### story
{story}

### question
{question}

### story ending
{ending}

requirments:
Please edit only the necessary parts of the original story to make it align logically with the essence of the ending and address the question. Avoid directly including or restating the ending's exact words in the revised story. Incorporate relevant elements from the question into the story, while minimizing changes to retain the original tone and style. Ensure the revised story flows coherently and connects naturally to both the question and the implied outcome of the ending.

note: The question has a causal direction (forwards or backwards). For the forwards direction, the question elicits the ending that is the more plausible effect (result) of the story context, whereas for the backwards direction, the question elicits the more plausible cause of the story context. Avoid making unnecessary changes to unrelated parts of the story.


### example1
Story: Ann had never been on an airplane. Her family was going on vacation next month to Florida. She was scared the plane was going to fall down. Her friend told her there was nothing to worry about.
Question: What outcome led to Amy having a lovely staycation at home?
story ending: She was frightened so much that she cancelled her vacation.
Edited Story: Ann had never been on an airplane. Her family was going on vacation next month to Florida. She was scared the plane was going to fall down. Her friend told her there was nothing to worry about, and Amy had a lovely staycation at home, because
question direction: backwards

### example2
Story: One day as I was coming home from work, my cat ran out the front door. I looked for her for hours that day. After a week of looking, I gave up hope that she would return. To my surprise, she came back to the front door after two whole weeks!
Question: If my cat was noticeably larger what was the likely reason?
story ending: She went out to mate, came back pregnant and had 8 pups.
Edited Story: One day as I was coming home from work, my cat ran out the front door. I looked for her for hours that day. After a week of looking, I gave up hope that she would return. To my surprise, she was noticeably larger and came back to the front door after two whole weeks!
question direction: forwards



Output format (use exactly this structure without additional symbols or capitalization variations):
edited story:<string of output story>

very important:
Avoid directly including or restating the ending's exact words in the revised story!!
"""


SELECT_ENDING_PROMPT = """
The following is a story beginning with four possible endings.

### story beginning
{story}

### ending 0:
{ending0}

### ending 1:
{ending1}

### ending 2:
{ending2}

### ending 3:
{ending3}

requirments:
Please select the ending that best fits the story beginning in terms of plot, tone, or logical progression. Provide the reason why this ending is the most suitable.

Output format (use exactly this structure without additional symbols or capitalization variations):
selected ending: <number of selected ending [0, 1, 2, or 3]>
explanation: <reason for selecting this ending>
"""

TEST_POSSIBLE_STORY_PROMPT = """
The following is a story beginning with four possible endings and a question.

### story beginning
{story}

### question
{question}

### ending 0:
{ending0}

### ending 1:
{ending1}

### ending 2:
{ending2}

### ending 3:
{ending3}

requirments:
Please select the ending that best fits the question based on plot consistency, tone alignment, and logical progression. Provide a clear explanation of why this ending is the most suitable, including specific references to the story and question. If possible, briefly explain why the other endings are less fitting.

Output format (use exactly this structure without additional symbols or capitalization variations):
selected ending: <number of selected ending [0, 1, 2, or 3]>
explanation: <reason for selecting this ending>
"""

SELECT_REASONING_TYPE_PROMPT = """
You are a reasoning expert tasked with analyzing questions based on their logical structure. Your goal is to determine whether a given question involves forward chaining (starting from known facts to infer outcomes) or backward chaining (starting from a desired goal to identify its causes or conditions). 

### Consider the following steps:
Analyze the direction of reasoning: 
- Does the question start with known facts and ask for potential outcomes? (Forward chaining)
- Or does it start with a goal or result and ask for possible causes or conditions? (Backward chaining)


Output format (use exactly this structure without additional symbols or capitalization variations):
ReasoningType: <Forward Chaining / Backward Chaining>

Now, classify the following question: 
{Question}
"""

FORWARD_CHAINING_EDIT_STORY_PROMPT = """
You are a skilled storyteller tasked with adapting a given story context to incorporate a hypothetical situation, ensuring the story logically leads to the specified conclusion without directly referencing or including any part of the conclusion in the narrative. The only change should be to subtly introduce the hypothetical situation from the question, and the ending should be represented as a placeholder @.

Instructions:
Minimize changes: Make as few edits as possible to the original story. Keep the tone, flow, and key details intact.
Only add the hypothetical situation: Ensure the hypothetical situation from the question is subtly implied in the context without introducing any extra information, detail, or expansion of the plot.
Avoid adding extra details: Do not add new information or explanations beyond the assumption implied by the question.
Avoid direct reference to the ending: Do not include or reword any part of the ending in the revised story.
Preserve original assumptions: Keep the assumptions and conditions from the original question intact, without altering or expanding upon them.

### example
Story: One day as I was coming home from work, my cat ran out the front door. I looked for her for hours that day. After a week of looking, I gave up hope that she would return. To my surprise, she came back to the front door after two whole weeks!
Question: If my cat was noticeably larger what was the likely reason?
story ending: She went out to mate, came back pregnant and had 8 pups.
Edited Story: One day as I was coming home from work, my cat ran out the front door. I looked for her for hours that day. After a week of looking, I gave up hope that she would return. To my surprise, she was noticeably larger and came back to the front door after two whole weeks!@

Output format (use exactly this structure without additional symbols or capitalization variations):
Revised Story: <Write the revised version here, incorporating the question and ensuring the ending is achieved.The ending is represented as the placeholder @> 

Now, revise the following story:

Story: {story}
Question: {question}
story ending: {ending}

very important:
Do not add any extra information or expand the story in any way.
Only introduce the hypothetical assumption from the question.
Represent the ending as @ in the revised story.
Keep the original question’s assumptions intact, but avoid explicitly stating or including the ending’s words.
"""

BACKWARD_CHAINING_EDIT_STORY_PROMPT = """
You are a creative writing assistant skilled in logical storytelling and backward reasoning. Your task is to adapt a given story context to incorporate a backward reasoning question naturally and align it with a specified result. The result should logically follow from the backward reasoning question and the story context but should be represented as a placeholder @ in the revised story.

Instructions:
Preserve the story structure: Make as few changes as possible to the original story, maintaining its characters, tone, and flow.
Subtly integrate the reasoning: Adjust the context minimally to logically imply the backward reasoning result without introducing explicit questions or significant rewrites.
Align with the backward reasoning condition and result(question): Adjust the context so the story logically leads to the result(question), the condition is represented as @.
Maintain coherence: Ensure the revised story flows naturally and remains consistent in tone and style.

### example
Story: Ann had never been on an airplane. Her family was going on vacation next month to Florida. She was scared the plane was going to fall down. Her friend told her there was nothing to worry about.
Question: What outcome led to Amy having a lovely staycation at home?
backward reasoning condition: She was frightened so much that she cancelled her vacation.
Edited Story: Ann had never been on an airplane. Her family was going on vacation next month to Florida. She was scared the plane was going to fall down. Her friend told her there was nothing to worry about, @, so Amy had a lovely staycation at home.

Output format (use exactly this structure without additional symbols or capitalization variations):
Revised Story: <Write the revised story here, incorporating the question and adding the placeholder @ for the question content.>

Now, revise the following story:

Story: {story}
Question: {question}
backward reasoning condition: {backward_reasoning_condition}

very important:
Do not add any extra information or expand the story in any way.
Only introduce the hypothetical assumption from the backward reasoning condition.
Represent the condition as @ in the revised story.
Avoid explicitly stating or including the backward reasoning condition’s words.
"""


SELECT_CONTENT_PROMPT = """
You are an advanced reasoning model tasked with completing a context by selecting the most appropriate option from a list of four candidates. Your goal is to ensure the chosen option logically and coherently fills the placeholder (@) in the context, maintaining consistency with the overall meaning and flow.

Instructions:
Carefully analyze the provided context and understand its meaning, tone, and structure.
Evaluate the four candidate options in terms of their logical fit and relevance to the context.
Choose the option that best fills the placeholder (@), ensuring the completed sentence or passage makes sense.
Provide a brief explanation of your choice to demonstrate your reasoning process.

Output format (use exactly this structure without additional symbols or capitalization variations):
selected content: <number of selected content [0, 1, 2, or 3]>
explanation: <reason for selecting this content>

Now, complete the following context:

### context
{context}

### content 0:
{content0}

### content 1:
{content1}

### content 2:
{content2}

### content 3:
{content3}

"""

GENERATE_STORY_CONTENT_PROMPT = """
Given the context below, please generate a sentence that fills in the missing part marked by "@", ensuring that it flows logically with the rest of the story while maintaining the original meaning and assumptions of the context.

Output format (use exactly this structure without additional symbols or capitalization variations):
generate story content: <generated content>
generated explanation: <reason for generating this content>

Now, complete the following context:

### context
{context}
"""
SELECT_CONTENT_PROMPT_HARD = """
You are an advanced reasoning model tasked with completing a context by selecting the most suitable option from a list of four candidates based on the given question.  Your goal is to ensure that the selected option answers the question and logically and coherently fills the placeholder (@) in the context while maintaining overall consistency in meaning and flow.

### Instructions:
1.  Carefully analyze the provided context to understand its meaning, tone, and structure.
2.  Evaluate the four candidate options for their logical fit and relevance to the context and question.
3.  Based on the context and question, select the option that best fills the placeholder (@), ensuring the completed context is meaningful and answers the question, either directly or indirectly.
4.  Provide a brief explanation of your choice to demonstrate your reasoning process.

### Output Format (use exactly this structure without any variations):
selected content: <number of the selected content [0, 1, 2, or 3]>
explanation: <reason for selecting this content>

Now, according to the question, complete the following context:

### Context:
{context}

### Question:
{question}

### Content 0:
{content0}

### Content 1:
{content1}

### Content 2:
{content2}

### Content 3:
{content3}
"""


GENERATE_INVERSE_OPTION_PROMPT = """
Based on the given context and question, along with the original ending, generate an alternative ending that conveys the opposite meaning or outcome of the original. Ensure that the new ending is logically consistent with the context and question, while reflecting the inverse of the original conclusion.

### example1
context: Bryant went skiing for the first time ever. His wife tried to make him go down the intermediate slope. Bryant got to the top of the hill and decided to walk back down. He got to the bottom and swore he'd never go up there again.
question: After going down the slope, Bryant went to see his wife who was currently not skiing and did what to forget skiing?
ending: He went back to the lodge and ordered Bloody Marys, one after another, trying to forget his anxiety and the slopes.
inverse ending: He went back to the lodge and ordered a cup of tea, savoring the calm and reflecting on how beautiful the slopes were.

Output format (use exactly this structure without additional symbols or capitalization variations):
inverse ending: <content of the inverse ending>

Now, generate an inverse ending based on the following content:

### context
{context}

### question
{question}

### ending:
{ending}
"""

DETECT_OPTION_CONTRACTION_PROMPT = """
Analyze the following sentence and determine if there is a contradiction within it. If the sentence contains a contradiction, return true. If the sentence does not contain a contradiction, return false.


Output format (use exactly this structure without additional symbols or capitalization variations):
Contradiction: <true or false>
Explanation: <reason for detecting the contradiction>

Now, detect if the following sentence contains a contradiction:

Sentence: {sentence}
"""

DETECT_QUESTION_CONTRACTION_PROMPT = """
Analyze the following two questions and determine if they contradict each other. If the two questions are contradictory, return true. If they are not contradictory, return false.


Output format (use exactly this structure without additional symbols or capitalization variations):
Contradiction: <true or false>
Explanation: <reason for detecting the contradiction>

Now, detect if the following two questions are contradictory:

Question 1: {question1}
Question 2: {question2}

"""

DETECT_OPTION_ISSUE_PROMPT = """
Based on the given two sentences, evaluate whether there are issues with their chronological order or logical coherence. If an issue exists, output 'true'; otherwise, output 'false'. Additionally, provide suggestions for revision to improve clarity and flow.

### sentences
{sentence}

Output format (use exactly this structure without additional symbols or capitalization variations):
issues: <true or false>
revision: <Suggestions for Revision without any explaination>
"""

EDIT_CONTEXT_PROMPT = """
You will be provided with:

1. A story segment labeled as **"context"**.
2. A hypothetical question about the story labeled as **"question"**.
3. A possible outcome related to the question labeled as **"answer"**.

Your task is to integrate the **"question"** into the **"context"** as a declarative narrative element, ensuring that the storyline reflects the hypothetical scenario described in the question without including the specific details from the **"answer"**. While maintaining the overall coherence of the context, ensure that the **"question"** is rephrased as a statement of past or ongoing events.

Guidelines:

- The **"question"** should not appear as a question but as part of the narrative flow.
- Avoid directly incorporating any content from the **"answer"** into the revised context.
- It's acceptable for the resulting story to have slight discontinuities or gaps, as it will serve as the foundation for a multiple-choice question where the **"answer"** will represent the most likely outcome.
- Make the minimum necessary changes to the **"context"** to accomplish the task.

Example Input:
**context:** Gus got into his car to pull out of his driveway. He looked behind him to back up but he could not see. There was too much snow on his window. Gus got out of the car and brushed it off.
**question:** Which outcome is the most likely precursor of Gus needing to go to urgent care for frostbite?
**answer:** he realized his hand was freezing.

Example Output:
Gus got into his car to pull out of his driveway. He looked behind him to back up but couldn't see anything. There was too much snow on his window. Gus got out of the car and brushed the snow off. Later, he had to go to urgent care.

**Context**:{context}

**Question**: {question}

**Answer**: {answer}

**Output**: ""
"""


OPTION_OPTIMIZATION_VERIFICATION_PROMPT = """
Task Description:
You will be given a short story context(5-6 sentences), followed by two questions and four options. Your goal is to carefully analyze the context and both question1 and question2 through reasoning. Then, among the four options, select the one that:

1. Correctly aligns with both question1 and question2.
2. Is what's most likely to happen in the story based on the given story context.
If none of the options fully satisfy both questions, select the best possible choice given the available information.

Input Format:
context: A short story context (5-6 sentences) providing information
question1: A hypothetical question that explores an alternative possibility within the story
question2: A factual question based on details from the story
options: Four possible events

Output Format(use exactly this structure without additional symbols or capitalization variations):
answer: Respond with a single integer (0,1,2, or 3), representing the correct answer choice
explanation: Reason for selecting this content

context: 
{context}
question1: 
{question1}
question2: 
{question2}
options: 
{options}

"""

SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_PROMPT = """
### Instruction:
You are given a story context, a hypothetical question about the story, and four possible event options. Your task is to analyze the context, understand the hypothetical condition in the question, and infer which of the four options best aligns with both the question's assumption and the logical flow of the story.

### Output Format(use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Input:
context: 
{context}
question: 
{question}
options:
0: {option0},
1: {option1},
2: {option2},
3: {option3}

"""

SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_COT_PROMPT = """
### Instruction:
You are given a story context, a hypothetical question about the story, and four possible event options. Your task is to analyze the context, understand the hypothetical condition in the question, and infer which of the four options best aligns with both the question's assumption and the logical flow of the story.

### Output Format(use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Input:
context: 
{context}
question: 
{question}
options:
0: {option0},
1: {option1},
2: {option2},
3: {option3}

Let's think step by step.
"""
SELECT_CONTENT_WITH_EXPLICIT_IMPLICIT_SETTING_ONE_SHOT_PROMPT = """
### Instruction:
You are given a story context, a hypothetical question about the story, and four possible event options. Your task is to analyze the context, understand the hypothetical condition in the question, and infer which of the four options best aligns with both the question's assumption and the logical flow of the story.

### Output Format(use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Example:
context: My neighbor called me over to have coffee. I was excited because I really needed it. When I got there he poured me a cup. I let it cool for a while in my hand. My neighbor seemed to need someone to talk to.
question:Which answer tells you that the neighbor did not clean the cup?
options: 
0: "When I went to take a sip I saw it had a roach in it and spit it out. It turned out my neighbor had some terrible news to share about their mother.",
1: "When I went to take a sip I saw it had a roach in it and spit it out. It turned out my neighbor had no terrible news to share about their mother.",
2: "When I went to take a sip, I saw it was clean and drank it. It turned out my neighbor had some terrible news to share about their mother.",
3: "When I went to take a sip I saw it had a roach in it and spit it out. I tasted the coffee and it was so delicious, I ended up drinking two cups!"

answer: 0
explanation: Option 0 shows that there is a roach in the cup, implying that it was not cleaned. In addition, option 0 also indicates that the neighbor has bad news that the neighbor wants to share. It is consistent with the context that the neighbor wants to find someone to talk to, so option 0 is the most consistent answer.

Now complete the following task:
### Input:
context: 
{context}
question: 
{question}
options:
0: {option0},
1: {option1},
2: {option2},
3: {option3}

"""
SELECT_CONTENT_WITH_IMPLICIT_SETTING_PROMPT = """
### Instruction:
You are given a story context and four possible event options. Your task is to analyze the context and infer which of the four options best aligns with the logical flow of the story.

### Output Format(use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Input:
context: 
{context}
options:
0: {option0},
1: {option1},
2: {option2},
3: {option3}

"""

SELECT_CONTENT_WITH_EXPLICIT_SETTING_PROMPT = """
### Instruction:
You are given a story context, a question about the story, and four possible event options. Your task is to analyze the context, understand the condition in the question, and infer which of the four options best aligns with both the question's content and the logical flow of the story.

### Output Format(use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Input:
context: 
{context}
question: 
{question}
options:
0: {option0},
1: {option1},
2: {option2},
3: {option3}

"""

SELECT_CONTENT_WITH_DOUBLE_EXPLICIT_SETTING_PROMPT = """
### Instruction:
You are given a story context, a hypothetical question about the story, a factual question based on details from the story, and four possible event options. Your task is to analyze the context, understand the hypothetical condition in the first question, consider the details in the second question, and infer which of the four options best aligns with both the content of the two questions and the logical flow of the story.

### Output Format(use exactly this structure without additional symbols or capitalization variations):
answer: <Respond with a single integer (0,1,2, or 3), representing the correct answer choice>
explanation: <Reason for selecting this option>

### Input:
context: 
{context}
question1: 
{question1}
question2: 
{question2}
options:
0: {option0},
1: {option1},
2: {option2},
3: {option3}
"""

TRANSLATE_PROMPT = """
请把下面六个英文内容翻译成中文：
content1:
{content1}
content2:
{content2}
content3:
{content3}
content3:
{content4}
content3:
{content5}
content3:
{content6}

输出格式（请严格遵循以下格式）：
answer1: <content1的翻译结果>
answer2: <content2的翻译结果>
answer3: <content3的翻译结果>
answer4: <content4的翻译结果>
answer5: <content5的翻译结果>
answer6: <content6的翻译结果>
"""