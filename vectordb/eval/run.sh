set -e

python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
python3 run.py --embeddings_csv=gs://datcom-nl-models/embeddings_medium_2024_03_14_16_38_53.ft_final_v20230717230459.all-MiniLM-L6-v2.csv "$@"
