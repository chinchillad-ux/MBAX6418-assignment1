# MBAX6418 Assignment 1

Binary sentiment classification of Amazon Gift Card reviews using an OpenAI-compatible LLM endpoint.

The classifier uses each review's `title` and `text` to predict `POSITIVE` or `NEGATIVE`. For evaluation, ratings 1–3 map to `NEGATIVE` and ratings 4–5 map to `POSITIVE`.

## Setup

1. Download `Gift_Cards.jsonl.gz` into this directory.
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure the endpoint. Do not commit the API key:

   ```bash
   export OPENAI_BASE_URL="http://dobolyi.com:9001/v1"
   export OPENAI_API_KEY="your-key"
   export OPENAI_MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
   ```

## Run

Process 100 reviews:

```bash
python classify.py --limit 100 --out predictions.csv
```

The script streams the compressed JSONL dataset, writes per-review predictions to CSV, and reports accuracy, a confusion matrix, precision, recall, and F1 score.

## Data

The dataset and generated prediction files are excluded from Git because they are downloaded/generated artifacts.