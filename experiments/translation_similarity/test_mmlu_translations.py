import os
from nltk.translate.chrf_score import sentence_chrf
from rouge_score import rouge_scorer
import pandas as pd
import sys
from tqdm import tqdm
from time import sleep

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))

from google_translate import translate_text

rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

input_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/mmlu_clinical_za'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/translation_similarity'))

sections = [
    'clinical_knowledge',
    'college_medicine',
]

splits = [
    'dev',
    'test',
    'val',
]

file_prefixes = []
for section in sections:
    for split in splits:
        file_prefixes.append(f"{section}_{split}")

language_suffixes = {
    'Afrikaans': '_af',
    'Zulu': '_zu',
    'Xhosa': '_xh',
}

for language, suffix in tqdm(language_suffixes.items()):
    lang_code = suffix[1:]

    files = []
    translations = []
    originals = []
    google_translates = []
    backtranslations = []
    google_translate_backtranslations = []
    rouge1_translation_vs_original = []
    rouge1_translation_vs_google_translate = []
    rouge1_backtranslation_vs_original = []
    rouge1_backtranslation_vs_google_translate_backtranslation = []
    rougeL_translation_vs_original = []
    rougeL_translation_vs_google_translate = []
    rougeL_backtranslation_vs_original = []
    rougeL_backtranslation_vs_google_translate_backtranslation = []
    chrF_translation_vs_original = []
    chrF_translation_vs_google_translate = []
    chrF_backtranslation_vs_original = []
    chrF_backtranslation_vs_google_translate_backtranslation = []

    for file_prefix in tqdm(file_prefixes):
        this_file_path = os.path.join(input_dir, f'{file_prefix}{suffix}.csv')
        this_file = pd.read_csv(this_file_path, header=None, delimiter=';')  # translations are delimited with semicolons
        english_file = pd.read_csv(os.path.join(input_dir, f'{file_prefix}.csv'), header=None)
        this_file_name = this_file_path[this_file_path.rfind('/')+1:]
        for index, row in tqdm(this_file.iterrows()):
            files.append(this_file_name)
            this_line = ' '.join([row.iloc[i] for i in range(5)])
            english_line = ' '.join([english_file.iloc[index, i] for i in range(5)])

            translations.append(this_line)
            originals.append(english_line)

            # Keep trying until success
            while True:
                try:
                    google_translation = translate_text(lang_code, english_line)
                    google_translation_backtranslation = translate_text(lang_code, google_translation, backtranslate=True)
                    backtranslation = translate_text(lang_code, this_line, backtranslate=True)
                    break
                except Exception as e:
                    print("An exception occurred:", e)
                    sleep(1)

            google_translates.append(google_translation)
            backtranslations.append(backtranslation)
            google_translate_backtranslations.append(google_translation_backtranslation)

            rouge_scores_translation_vs_original = rouge_scorer_obj.score(this_line, english_line)
            rouge_scores_translation_vs_google_translate = rouge_scorer_obj.score(this_line, google_translation)
            rouge_scores_backtranslation_vs_original = rouge_scorer_obj.score(backtranslation, english_line)
            rouge_scores_backtranslation_vs_google_translate_backtranslation = rouge_scorer_obj.score(backtranslation, google_translation_backtranslation)

            rouge1_translation_vs_original.append(rouge_scores_translation_vs_original['rouge1'].fmeasure)
            rouge1_translation_vs_google_translate.append(rouge_scores_translation_vs_google_translate['rouge1'].fmeasure)
            rouge1_backtranslation_vs_original.append(rouge_scores_backtranslation_vs_original['rouge1'].fmeasure)
            rouge1_backtranslation_vs_google_translate_backtranslation.append(rouge_scores_backtranslation_vs_google_translate_backtranslation['rouge1'].fmeasure)
            rougeL_translation_vs_original.append(rouge_scores_translation_vs_original['rougeL'].fmeasure)
            rougeL_translation_vs_google_translate.append(rouge_scores_translation_vs_google_translate['rougeL'].fmeasure)
            rougeL_backtranslation_vs_original.append(rouge_scores_backtranslation_vs_original['rougeL'].fmeasure)
            rougeL_backtranslation_vs_google_translate_backtranslation.append(rouge_scores_backtranslation_vs_google_translate_backtranslation['rougeL'].fmeasure)

            chrF_scores_translation_vs_original = sentence_chrf(this_line.split(), english_line.split())
            chrF_scores_translation_vs_google_translate = sentence_chrf(this_line.split(), google_translation.split())
            chrF_scores_backtranslation_vs_original = sentence_chrf(backtranslation.split(), english_line.split())
            chrF_scores_backtranslation_vs_google_translate_backtranslation = sentence_chrf(backtranslation.split(), google_translation_backtranslation.split())

            chrF_translation_vs_original.append(chrF_scores_translation_vs_original)
            chrF_translation_vs_google_translate.append(chrF_scores_translation_vs_google_translate)
            chrF_backtranslation_vs_original.append(chrF_scores_backtranslation_vs_original)
            chrF_backtranslation_vs_google_translate_backtranslation.append(chrF_scores_backtranslation_vs_google_translate_backtranslation)

    lang_df = pd.DataFrame(
        {
            'source_file': files,
            'translation': translations,
            'original_line': originals,
            'google_translation': google_translates,
            'backtranslation': backtranslations,
            'google_translation_backtranslation': google_translate_backtranslations,
            'rouge1_translation_vs_original': rouge1_translation_vs_original,
            'rouge1_translation_vs_google_translate': rouge1_translation_vs_google_translate,
            'rouge1_backtranslation_vs_original': rouge1_backtranslation_vs_original,
            'rouge1_backtranslation_vs_google_translate_backtranslation': rouge1_backtranslation_vs_google_translate_backtranslation,
            'rougeL_translation_vs_original': rougeL_translation_vs_original,
            'rougeL_translation_vs_google_translate': rougeL_translation_vs_google_translate,
            'rougeL_backtranslation_vs_original': rougeL_backtranslation_vs_original,
            'rougeL_backtranslation_vs_google_translate_backtranslation': rougeL_backtranslation_vs_google_translate_backtranslation,
            'chrF_translation_vs_original': chrF_translation_vs_original,
            'chrF_translation_vs_google_translate': chrF_translation_vs_google_translate,
            'chrF_backtranslation_vs_original': chrF_backtranslation_vs_original,
            'chrF_backtranslation_vs_google_translate_backtranslation': chrF_backtranslation_vs_google_translate_backtranslation,
        }
    )

    lang_df.to_csv(os.path.join(output_dir, f'mmlu_translation_similarity_{lang_code}.csv'), index=False)

    # Get averages of each columns
    numerical_df = lang_df.select_dtypes(include='number')
    means = numerical_df.mean()
    means_df = pd.DataFrame(means).T
    means_df.to_csv(os.path.join(output_dir, f'mmlu_translation_similarity_{lang_code}_averages.csv'), index=False)
