import os
import json
from openai import OpenAI
from tqdm import tqdm
from time import sleep
import io
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Get the current date and time
now = datetime.datetime.now()

# Format the date and time as a string
timestamp = now.strftime("%Y%m%d_%H%M%S")

USE_MOST_RECENT_8_MODELS = True


def display_options(options):
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")


def get_selection(options):
    while True:
        display_options(options)
        selection = input("Please enter a comma-separated list of numbers corresponding to your desired fine-tuned models to evaluate (e.g. \"1,2,3\"): ")
        try:
            selected_indices = [int(num) for num in selection.split(",")]
            selected_strings = [options[idx - 1] for idx in selected_indices if 1 <= idx <= len(options)]
            print("You have selected:")
            for string in selected_strings:
                print(f"- {string}")
            confirm = input("Is this correct? (yes/no): ").strip().lower()
            if confirm == 'yes':
                return selected_strings
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid numbers corresponding to the options.")
        print("Retrying...")


# Define response-to-correctness functions
def check_mc_answer(custom_id, generation):
    parsed_gen = generation.strip().replace('(', ''). replace(')', '').upper()
    return len(parsed_gen) > 0 and parsed_gen[0] == custom_id[-1]  # answer is stored in last number of custom_id


def check_winogrande_answer(custom_id, generation):
    correct_number = custom_id[-1]  # answer is stored in the last character of the custom_id
    incorrect_number = str(3 - int(correct_number))  # maps 1 to 2 and 2 to 1
    correct = correct_number in generation and incorrect_number not in generation
    return correct


with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../config.json')), 'r') as fp:
    config = json.load(fp)
    openai_api_key = config['openai_key']
    openai_org = config['openai_org']

eval_file_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_batches/'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/gpt_35_cross_lingual/'))
client = OpenAI(api_key=openai_api_key)

# Prepare template
eval_file_path = os.path.join(eval_file_dir, "gpt_style_batch_evaluation_template.jsonl")
with open(eval_file_path, 'r') as fp:
    eval_template = fp.read()

# Sort from newest to oldest by reversing
available_tuned_models = list(reversed([model.id for model in client.models.list() if model.owned_by == openai_org and ':ckpt-step-' not in model.id]))
if len(available_tuned_models) == 0:
    print("No fine-tuned models available. Please check your OpenAI key and org in the config.json file.")
else:
    if not USE_MOST_RECENT_8_MODELS:
        desired_models = sorted(get_selection(available_tuned_models))
    else:
        desired_models = sorted(available_tuned_models[:8])
    desired_models += ['gpt-3.5-turbo-1106', 'gpt-4-turbo', 'gpt-4o']  # Add baseline models
    batch_ids = []
    input_ids = []
    batch_to_model = {}
    print("Creating batches...")
    for desired_model in tqdm(desired_models):
        # In case of too many requests per minute, try again
        while True:
            try:
                # Replace placeholder with actual model name
                this_eval = io.BytesIO(eval_template.replace("<|MODEL|>", desired_model).encode())

                this_id = client.files.create(
                    file=this_eval,
                    purpose="batch"
                ).id
                batch_id = client.batches.create(
                    input_file_id=this_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h"
                ).id

                input_ids.append(this_id)
                batch_ids.append(batch_id)
                batch_to_model[batch_id] = desired_model
                print(f"Evaluation batch sent for model \"{desired_model}\"")

                break
            except Exception as e:
                print("An exception occurred:", e)
                sleep(60)

    # Wait for batches to finish
    while True:
        try:
            all_done = True
            for batch_id in batch_ids:
                this_batch = client.batches.retrieve(batch_id)
                if not this_batch.status == "completed":
                    all_done = False
                    print(f"Batch ID {batch_id} not yet completed. Progress: {this_batch.request_counts.completed}/{this_batch.request_counts.total}")
            if all_done:
                break
            else:
                print("Checking again in one minute...")
                sleep(60)
                    
        except Exception as e:
            print("An exception occurred:", e)
            sleep(60)

    print("All batches completed! Downloading the results and evaluating...")

    # Retrieve batch outputs
    while True:
        try:
            for batch_id in batch_ids:
                this_batch = client.batches.retrieve(batch_id)
                output_file_id = this_batch.output_file_id
                file_content = client.files.content(output_file_id)
                # Save the outputs
                with open(os.path.join(output_dir, f'generations_{batch_to_model[batch_id]}_{timestamp}.jsonl'), 'w') as fp:
                    fp.write(file_content.text)
            break

        except Exception as e:
            print("An exception occurred:", e)
            sleep(60)

    # Finally, it's time to get results!
    # Create map of every single generation
    complete_jsonl = []
    for model_name in desired_models:
        full_path = os.path.join(output_dir, f'generations_{model_name}.jsonl')
        with open(full_path, 'r') as fp:
            complete_jsonl.append(pd.read_json(fp, lines=True))

    complete_jsonl = pd.concat(complete_jsonl)
    complete_jsonl['generation'] = complete_jsonl['response'].apply(lambda x: x['body']['choices'][0]['message']['content'])
    generations_map = dict(zip(complete_jsonl['custom_id'], complete_jsonl['generation']))

    # Get and display MMLU performance
    sections = [
        'clinical_knowledge',
        'college_medicine',
    ]

    langs = [
        'en',
        'af',
        'zu',
        'xh',
    ]

    matrix = pd.DataFrame(
        data=0.0,
        index=desired_models,
        columns=langs
    )

    for this_model in desired_models:

        for lang in langs:
            total_score = 0
            q_cnt = 0

            for section in sections:
                # Skip evaluating college_medicine MMLU on college_medicine
                if section == 'college_medicine' and 'mmlu-' in this_model.lower():
                    continue

                # Construct the pattern
                pattern = re.compile(rf"{this_model}-on-{lang}-mmlu-{section}.*")

                # Filter keys
                matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

                print(len(matching_generations))

                for (c_id, gen) in matching_generations:
                    if check_mc_answer(c_id, gen):
                        total_score += 1
                    q_cnt += 1

            final_score = total_score / q_cnt
            matrix.at[this_model, lang] = round(final_score * 100, 1)

    # Create the heatmap
    plt.figure(figsize=(12, 8), dpi=100)  # Increase the figure size and resolution for HD
    ax = sns.heatmap(matrix, annot=matrix, cmap="Greens", cbar=False, annot_kws={"size": 16}, fmt='.1f')

    # Rotate the labels on the y-axis (left) to be horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)  # Increase y-axis label size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)  # Increase x-axis label size

    # Display the heatmap
    plt.tight_layout()
    plt.show()
    matrix.to_csv(os.path.join(output_dir, f'mmlu_{timestamp}.csv'))

    matrix = pd.DataFrame(
        data=0.0,
        index=desired_models,
        columns=langs
    )

    for this_model in desired_models:

        for lang in langs:
            total_score = 0
            q_cnt = 0

            # Construct the pattern
            pattern = re.compile(rf"{this_model}-on-{lang}-winogrande.*")

            # Filter keys
            matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]
            print(len(matching_generations))

            for (c_id, gen) in matching_generations:
                if check_winogrande_answer(c_id, gen):
                    total_score += 1
                q_cnt += 1

            final_score = total_score / q_cnt
            matrix.at[this_model, lang] = round(final_score * 100, 1)

    # Create the heatmap
    plt.figure(figsize=(12, 8), dpi=100)  # Increase the figure size and resolution for HD
    ax = sns.heatmap(matrix, annot=matrix, cmap="Greens", cbar=False, annot_kws={"size": 16}, fmt='.1f')

    # Rotate the labels on the y-axis (left) to be horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)  # Increase y-axis label size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)  # Increase x-axis label size

    # Display the heatmap
    plt.tight_layout()
    plt.show()
    matrix.to_csv(os.path.join(output_dir, f'winogrande_{timestamp}.csv'))

    matrix = pd.DataFrame(
        data=0.0,
        index=desired_models,
        columns=langs
    )

    for this_model in desired_models:

        for lang in langs:
            total_score = 0
            q_cnt = 0

            # Construct the pattern
            pattern = re.compile(rf"{this_model}-on-{lang}-belebele.*")

            # Filter keys
            matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]
            print(len(matching_generations))

            for (c_id, gen) in matching_generations:
                if check_mc_answer(c_id, gen):
                    total_score += 1
                q_cnt += 1

            final_score = total_score / q_cnt
            matrix.at[this_model, lang] = round(final_score * 100, 1)

    # Create the heatmap
    plt.figure(figsize=(12, 8), dpi=100)  # Increase the figure size and resolution for HD
    ax = sns.heatmap(matrix, annot=matrix, cmap="Greens", cbar=False, annot_kws={"size": 16}, fmt='.1f')

    # Rotate the labels on the y-axis (left) to be horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)  # Increase y-axis label size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)  # Increase x-axis label size

    # Display the heatmap
    plt.tight_layout()
    plt.show()
    matrix.to_csv(os.path.join(output_dir, f'belebele_{timestamp}.csv'))
