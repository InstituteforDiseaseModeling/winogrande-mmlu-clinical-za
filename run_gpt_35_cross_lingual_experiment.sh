if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_evaluations>"
    exit 1
fi
num_evaluations=$1
echo "Fine-tuning and running $num_evaluations evaluations..."
python3 experiments/gpt_35_cross_lingual/create_fine_tuning_jobs.py
for ((i=1; i<=num_evaluations; i++))
do
    python3 experiments/gpt_35_cross_lingual/evaluate_fine_tuned_models.py
done
