# Data

This directory contains five folders:

`belebele_za/`:
- Contains [Belebele](https://github.com/facebookresearch/belebele) from Meta in English, Afrikaans, Zulu, and Xhosa.
  - `belebele.jsonl`: Contains Belebele in English
  - `belebele_af.jsonl`: Contains Belebele in Afrikaans
  - `belebele_xh.jsonl`: Contains Belebele in Xhosa
  - `belebele_zu.jsonl`: Contains Belebele in Zulu

`evaluation_batches/`:
- Contains GPT-style batch jobs readable by the [Batch API](https://platform.openai.com/docs/guides/batch).
  - `gpt_style_batch_evaluation_template.jsonl`: Contains all the questions required to evaluate a model on Belebele, MMLU clinical knowledge and college medicine test, and Winogrande test, each of which are given in all of English, Afrikaans, Zulu, and Xhosa. The `<|MODEL|>` string within can be substituted to evaluate on any desired GPT model by name.
  - `gpt_style_batch_evaluation_template_test.jsonl`: Contains 1 question from each of Belebele, MMLU (both sections test), and Winogrande test, in each of English, Afrikaans, Zulu, and Xhosa, for a total of 16 questions. The `<|MODEL|>` string within can be substituted to evaluate on any desired GPT model by name.

`gpt_fine_tuning_datasets/`:
- Contains GPT-style fine-tuning jobs readable by the [Fine-Tuning API](https://platform.openai.com/docs/guides/fine-tuning).
  - `mmlu_college_medicine_af.jsonl`: Dataset used to fine-tune a GPT model in MMLU college medicine in Afrikaans.
  - `mmlu_college_medicine_en.jsonl`: Dataset used to fine-tune a GPT model in MMLU college medicine in English.
  - `mmlu_college_medicine_xh.jsonl`: Dataset used to fine-tune a GPT model in MMLU college medicine in Xhosa.
  - `mmlu_college_medicine_zu.jsonl`: Dataset used to fine-tune a GPT model in MMLU college medicine in Zulu.
  - `winogrande_train_s_af.jsonl`: Dataset used to fine-tune a GPT model in Winogrande Train-S in Afrikaans.
  - `winogrande_train_s_en.jsonl`: Dataset used to fine-tune a GPT model in Winogrande Train-S in English.
  - `winogrande_train_s_xh.jsonl`: Dataset used to fine-tune a GPT model in Winogrande Train-S in Xhosa.
  - `winogrande_train_s_zu.jsonl`: Dataset used to fine-tune a GPT model in Winogrande Train-S in Zulu.

`mmlu_clinical_za/`:
- Contains MMLU college medicine and clinical knowledge sections human-translated into Afrikaans, Xhosa, and Zulu (as well as the English originals). These are the files directly given by [Translated.com](https://Translated.com). **Note that the non-English language files have semicolon delimiters.**
  - `{section}_{split}{lang_suffix}`: Portion of MMLU where `{section}` is one of `clinical_knowledge` or `college_medicine`, `{split}` is one of `dev`, `test`, or `val`, and `{lang_suffix}` is empty for English, `_af` for Afrikaans, `_xh` for Xhosa, and `_zu` for Zulu.

`winogrande_za/`:
- Contains Winogrande Train-S, Test, and Dev splits human-translated into Afrikaans, Xhosa, and Zulu (as well as the English original). The original JSONL files have been merged into CSV files with a `split` column to denote the split. The non-English language files also contain the English originals.
  - `winogrande{lang_suffix}`: Winogrande where `{lang_suffix}` is empty for English, `_af` for Afrikaans, `_xh` for Xhosa, and `_zu` for Zulu.
