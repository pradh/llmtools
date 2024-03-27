set -e

python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
python3 run.py \
    --store_embeddings=gs://datcom-nl-models/embeddings_medium_2024_03_14_16_38_53.ft_final_v20230717230459.all-MiniLM-L6-v2.csv \
    --test_embeddings=gs://datcom-nl-models/embeddings_undata_2024_03_20_11_01_12.ft_final_v20230717230459.all-MiniLM-L6-v2.csv \
    "$@"
