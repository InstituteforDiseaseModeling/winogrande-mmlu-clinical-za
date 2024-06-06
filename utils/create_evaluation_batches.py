import pandas as pd
import os
import json

MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

suffix_map = {
    'en': '',
    'af': '_af',
    'xh': '_xh',
    'zu': '_zu',
}

languages = {
    'en': 'English',
    'af': 'Afrikaans',
    'zu': 'Zulu',
    'xh': 'Xhosa',
}

sections = [
    'clinical_knowledge',
    'college_medicine',
]

input_dir_mmlu = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../data/mmlu_clinical_za'))
input_dir_winogrande = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../data/winogrande_za'))
input_dir_belebele = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../data/belebele_za'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../data/evaluation_batches'))

# Ensure the same Winogrande dev questions are used no matter the order of the qIDs in the files
winogrande_en = pd.read_csv(os.path.join(input_dir_winogrande, 'winogrande.csv'))
dev_qIDs = set(winogrande_en[winogrande_en['Split'] == 'dev'].sample(n=5, random_state=42)['qID'].tolist())

belebele_correct_answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

belebele_base_prompt = """Given the following passage, query, and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.
###
Passage:
{passage}
###
Query:
{query}
###
Choices:
(A) {a}
(B) {b}
(C) {c}
(D) {d}
###
Answer:"""

model_placeholder = '<|MODEL|>'

batch = []

for lang_code, suffix in suffix_map.items():

    # Add MMLU questions (5-shot)
    for section in sections:
        test_section = pd.read_csv(
            os.path.join(
                input_dir_mmlu,
                f'{section}_test{suffix}.csv'),
            delimiter=(',' if lang_code == 'en' else ';'),
            header=None)

        dev_section = pd.read_csv(
            os.path.join(
                input_dir_mmlu,
                f'{section}_dev{suffix}.csv'),
            delimiter=(',' if lang_code == 'en' else ';'),
            header=None)

        for index, row in test_section.iterrows():
            this_prompt = f"""The following are multiple choice questions (with answers) about {section.replace("_", " ")}.

Question 1: {dev_section.iloc[0, 0]}
A. {dev_section.iloc[0, 1]}
B. {dev_section.iloc[0, 2]}
C. {dev_section.iloc[0, 3]}
D. {dev_section.iloc[0, 4]}
Answer: {dev_section.iloc[0, 5]}

Question 2: {dev_section.iloc[1, 0]}
A. {dev_section.iloc[1, 1]}
B. {dev_section.iloc[1, 2]}
C. {dev_section.iloc[1, 3]}
D. {dev_section.iloc[1, 4]}
Answer: {dev_section.iloc[1, 5]}

Question 3: {dev_section.iloc[2, 0]}
A. {dev_section.iloc[2, 1]}
B. {dev_section.iloc[2, 2]}
C. {dev_section.iloc[2, 3]}
D. {dev_section.iloc[2, 4]}
Answer: {dev_section.iloc[2, 5]}

Question 4: {dev_section.iloc[3, 0]}
A. {dev_section.iloc[3, 1]}
B. {dev_section.iloc[3, 2]}
C. {dev_section.iloc[3, 3]}
D. {dev_section.iloc[3, 4]}
Answer: {dev_section.iloc[3, 5]}

Question 5: {dev_section.iloc[4, 0]}
A. {dev_section.iloc[4, 1]}
B. {dev_section.iloc[4, 2]}
C. {dev_section.iloc[4, 3]}
D. {dev_section.iloc[4, 4]}
Answer: {dev_section.iloc[4, 5]}

Now, given the following question and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.

Question: {row.iloc[0]}
A. {row.iloc[1]}
B. {row.iloc[2]}
C. {row.iloc[3]}
D. {row.iloc[4]}
Answer:
"""

            batch.append(
                {"custom_id": f"{model_placeholder}-on-{lang_code}-mmlu-{section}-{index}-answer-{row.iloc[5]}",
                 "method": "POST",
                 "url": "/v1/chat/completions",
                 "body": {"model": model_placeholder,
                          "messages": [{"role": "user",
                                        "content": this_prompt}],
                          "max_tokens": MAX_TOKENS,
                          "temperature": TEMPERATURE,
                          "top_p": TOP_P}}
            )

    # Add Winogrande Questions (5-shot)
    language = languages[lang_code]
    this_winogrande = pd.read_csv(os.path.join(input_dir_winogrande, f'winogrande{suffix}.csv'))

    # Get dev and test sets
    this_dev_set = this_winogrande[this_winogrande['qID'].isin(dev_qIDs)].sort_values(by=['qID']).reset_index(drop=True)
    this_test_set = this_winogrande[this_winogrande['Split'] == 'test'].sort_values(by=['qID']).reset_index(drop=True)
    for index, row in this_test_set.iterrows():
        this_prompt = f"""The following are sentences that are missing a word or a few words (denoted with an underscore), each followed by two options to fill in the missing word or words. The correct option is given for each sentence:

Sentence 1: {this_dev_set.iloc[0][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[0][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[0][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[0]["Answer"]}

Sentence 2: {this_dev_set.iloc[1][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[1][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[1][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[1]["Answer"]}

Sentence 3: {this_dev_set.iloc[2][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[2][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[2][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[2]["Answer"]}

Sentence 4: {this_dev_set.iloc[3][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[3][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[3][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[3]["Answer"]}

Sentence 5: {this_dev_set.iloc[4][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[4][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[4][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[4]["Answer"]}

Now, given the following sentence and options, output only the number corresponding to the correct option. Do not add any explanation.

Sentence: {this_test_set.iloc[index][f"{language} Sentence"]}
Option1: {this_test_set.iloc[index][f"{language} Option 1"]}
Option2: {this_test_set.iloc[index][f"{language} Option 2"]}
Correct Option:
"""

        batch.append(
            {"custom_id": f"{model_placeholder}-on-{lang_code}-winogrande-{index}-answer-{this_test_set.iloc[index]['qID'][-1]}",
             "method": "POST",
             "url": "/v1/chat/completions",
             "body": {"model": model_placeholder,
                      "messages": [{"role": "user",
                                    "content": this_prompt}],
                      "max_tokens": MAX_TOKENS,
                      "temperature": TEMPERATURE,
                      "top_p": TOP_P}}
        )

    # Add Belebele questions (0-shot)
    with open(os.path.join(input_dir_belebele, f'belebele{suffix}.jsonl'), 'r') as fp:
        this_belebele = pd.read_json(fp, lines=True)
        this_belebele = this_belebele.sort_values(by=['link', 'question_number'])  # Ensure consistent order across langs

    for index, row in this_belebele.iterrows():
        this_prompt = belebele_base_prompt \
            .replace("{passage}", row['flores_passage']) \
            .replace("{query}", row['question']) \
            .replace("{a}", row['mc_answer1']) \
            .replace("{b}", row['mc_answer2']) \
            .replace("{c}", row['mc_answer3']) \
            .replace("{d}", row['mc_answer4'])
        batch.append(
            {
                "custom_id": f"{model_placeholder}-on-{lang_code}-belebele-{index}-answer-{belebele_correct_answer_map[row['correct_answer_num']]}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model_placeholder,
                         "messages": [{"role": "user",
                                       "content": this_prompt}],
                         "max_tokens": MAX_TOKENS,
                         "temperature": TEMPERATURE,
                         "top_p": TOP_P}}
        )

# Write the sampled data to a JSONL file
with open(os.path.join(output_dir, 'gpt_style_batch_evaluation_template.jsonl'), 'w') as f:
    for item in batch:
        f.write(json.dumps(item) + '\n')
