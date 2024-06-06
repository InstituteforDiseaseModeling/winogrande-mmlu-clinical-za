# Experiments

This directory contains three folders, the same exact folders as `../results/`, which correspond to where
the results of the experiments go (e.g. `translation_similarity/` experiment results go in `../results/translation_similarity/`):

`gpt_35_cross_lingual/`:
- Contains Python scripts to use the OpenAI fine-tuning API and batch API to fine-tune GPT models and evaluate them.
  - `create_fine_tuning_jobs.py`: Creates 8 fine-tuning jobs, 4 to tune on MMLU's college medicine section in each of Afrikaans, English, Xhosa, and Zulu and 4 to tune on Winogrande Train-S in each of Afrikaans, English, Xhosa, and Zulu. The script will not terminate until all fine-tuning jobs have finished.
  - `evaluate_fine_tuned_models.py`: Takes the 8 fine-tuned models and creates batches to evaluate them on all of MMLU college medicine + clinical knowledge (test sets), Winogrande Test, and Belebele, in each of Afrikaans, English, Xhosa, and Zulu. This script will not terminate until all batches have finished and will save the results as CSV/JSONL files to `../results/gpt_35_cross_lingual/`. Note that this script also evaluates GPT models out of the box (GPT-3.5, GPT-4, GPT-4o), adding to the results of the fine-tuned models.

`out_of_the_box_performance/`:
- Contains Jupyter notebooks to evaluate a Hugging Face model (denoted by the filename of the notebook).
  - `Evaluate_{model_name}.ipynb`: Tests `{model_name}` on all of MMLU college medicine + clinical knowledge (test sets), Winogrande Test, and Belebele, in each of Afrikaans, English, Xhosa, and Zulu.

`translation_similarity/`:
- Contains Python scripts to use the Google Translate API to obtain translation similarity scores across the MMLU and Winogrande human-translations.
  - `test_{benchmark}_translations.py`: Runs translation similarity tests on the translations for `{benchmark}` (MMLU or Winogrande).

