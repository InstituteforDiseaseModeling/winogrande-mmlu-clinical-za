import os
import json
from openai import OpenAI
from time import sleep
BASE_MODEL = 'gpt-3.5-turbo-1106'
SEED = 42
SIMULTANEOUS_FINE_TUNING_LIMIT = 3


with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../config.json')), 'r') as fp:
    config = json.load(fp)
    openai_api_key = config['openai_key']

input_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/gpt_fine_tuning_datasets'))

client = OpenAI(api_key=openai_api_key)

for tuning_file in sorted(os.listdir(input_dir)):
    input_path = os.path.join(input_dir, tuning_file)

    tuning_file_pieces = tuning_file.split('_')
    suffix = f"{tuning_file_pieces[0]}-{tuning_file_pieces[-1][:2]}"

    # Wait for fine-tuning jobs to finish before starting other ones as to not exceed the API limit of having a certain
    # number of jobs running simultaneously
    while True:
        in_progress_cnt = 0
        for job in client.fine_tuning.jobs.list():
            if job.status in {'running', 'validating_files', 'queued'}:
                in_progress_cnt += 1
        if in_progress_cnt < SIMULTANEOUS_FINE_TUNING_LIMIT:
            break
        else:
            print(f"Too many consecutive fine-tuning jobs ({in_progress_cnt}). Waiting for 1 minute...")
            sleep(60)

    # In case of too many requests per minute, try again
    while True:
        try:
            this_id = client.files.create(
                file=open(input_path, "rb"),
                purpose="fine-tune"
            ).id

            # Use automatic batch size, epochs, and learning rate multiplier
            client.fine_tuning.jobs.create(
                training_file=this_id,
                seed=SEED,
                model=BASE_MODEL,
                suffix=suffix
            )

            print(f"Fine-tuning job created with suffix \"{suffix}\"")

            break
        except Exception as e:
            print("An exception occurred:", e)
            print("Waiting for 1 minute...")
            sleep(60)

# Wait for completion
while True:
    in_progress_cnt = 0
    for job in client.fine_tuning.jobs.list():
        if job.status in {'running', 'validating_files', 'queued'}:
            in_progress_cnt += 1
    if in_progress_cnt == 0:
        break
    else:
        print(f"Waiting for fine-tuning jobs to finish...\nActive fine-tuning jobs: {in_progress_cnt}\nWaiting for 1 minute...")
        sleep(60)
