import json
import pandas as pd
import os

SYSTEM_MESSAGE = "You are a chatbot created by Ghamut and the Bill & Melinda Gates Foundation."

input_dir_mmlu = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../data/mmlu_clinical_za'))
input_dir_winogrande = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../data/winogrande_za'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../data/gpt_fine_tuning_datasets'))

mmlu_splits = ['dev', 'test', 'val']  # train on college medicine (full), test on clinical knowledge (test)

suffixes = {
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

for lang_code, suffix in suffixes.items():
    language = languages[lang_code]

    mmlu_full = []
    for split in mmlu_splits:
        mmlu_full.append(pd.read_csv(os.path.join(input_dir_mmlu, f'college_medicine_{split}{suffix}.csv'), delimiter=(',' if lang_code=='en' else ';'), header=None))

    mmlu_data = pd.concat(mmlu_full)

    winogrande_data = pd.read_csv(os.path.join(input_dir_winogrande, f'winogrande{suffix}.csv'))
    winogrande_data = winogrande_data[winogrande_data['Split'] == 'train_s'].sort_values(by=['qID']).reset_index(drop=True)  # Ensure consistent order across langs

    transformed_mmlu = []
    for index, row in mmlu_data.iterrows():
        this_prompt = f"""Given the following question and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.

Question: {row.iloc[0]}
A. {row.iloc[1]}
B. {row.iloc[2]}
C. {row.iloc[3]}
D. {row.iloc[4]}
Answer:
"""
        transformed_entry = {
            "messages": [
                {"role": "system",
                 "content": SYSTEM_MESSAGE},
                {"role": "user", "content": this_prompt},
                {"role": "assistant", "content": row.iloc[5]}
            ]
        }
        transformed_mmlu.append(transformed_entry)

    with open(os.path.join(output_dir, f'mmlu_college_medicine_{lang_code}.jsonl'), 'w') as f:
        for item in transformed_mmlu:
            f.write(json.dumps(item) + '\n')

    transformed_winogrande = []
    for index, row in winogrande_data.iterrows():
        this_prompt = f"""Given the following sentence that is missing a word or a few words (denoted with an underscore) and two options to fill in the missing word or words, output only the number corresponding to the correct option. Do not add any explanation.

Sentence: {winogrande_data.iloc[index][f"{language} Sentence"]}
Option1: {winogrande_data.iloc[index][f"{language} Option 1"]}
Option2: {winogrande_data.iloc[index][f"{language} Option 2"]}
Correct Option:
"""
        transformed_entry = {
            "messages": [
                {"role": "system",
                 "content": SYSTEM_MESSAGE},
                {"role": "user", "content": this_prompt},
                {"role": "assistant", "content": str(int(winogrande_data.iloc[index]['Answer']))}
            ]
        }
        transformed_winogrande.append(transformed_entry)

    with open(os.path.join(output_dir, f'winogrande_train_s_{lang_code}.jsonl'), 'w') as f:
        for item in transformed_winogrande:
            f.write(json.dumps(item) + '\n')
