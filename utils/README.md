# Utils

This directory contains helper scripts that support the experiments performed for the paper.

- `create_evaluation_batches`: Recreates the `../data/evaluation_batches/gpt_style_batch_evaluation_template.jsonl` file used as a template for sending evaluation batches to OpenAI's [Batch API](https://platform.openai.com/docs/guides/batch).
- `create_gpt_fine_tuning_datasets.py`: Recreates the fine-tuning datasets in `../data/gpt_fine_tuning_datasets/` used for fine-tuning GPT models with OpenAI's [Fine-Tuning API](https://platform.openai.com/docs/guides/fine-tuning).
- `google_translate.py`: Contains a utility function `translate_text()` for Google Translating text.
