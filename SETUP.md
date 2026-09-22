# Run and verify the submission

## View only — no setup or network

Download `results_dashboard.html` and open it in a browser. GitHub displays HTML source rather than executing the dashboard. The file embeds its data/styles/scripts; no API key is needed. The README screenshots provide an immediate interface preview on GitHub.

## Rebuild from saved output — no model calls

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_resources.py --data
# Read the NRC terms in the next section before downloading the lexicon.
python download_resources.py --lexicon
python build_dashboard.py
python write_report.py
```

The builder checks source identities against `balanced_sample.json` and the full dataset hash. `results_final.json` contains the final scored data; `final_raw.jsonl` contains verbatim model final answers and request metadata. The manifest was selected with a fixed seed from the whole source file. It also retains the earlier sampling-stage sentiment prompt; **the final inference prompt is `prompt.txt` and `results_final.json.metadata.system_prompt`**.

## Recompute the word-list scores — no model calls

Read the [NRC lexicon author page](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) and the terms in its download. The lexicon permits non-commercial research/education and prohibits redistribution. If the homepage is unavailable, the [author-hosted archive](https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip) includes the original README and terms. The download below is direct from the publisher, not a redistributed copy.

```bash
python download_resources.py --lexicon
python add_emotions.py
python build_dashboard.py
```

`resources/` is ignored by Git. A hash of the lexicon used is recorded in the final output. This implementation uses English word-level v0.92 associations, counts repeated exact lowercase tokens in title and body, ignores the two polarity categories, and sums only the eight emotion flags. It does not stem or resolve negation. Ties use alphabetical order (anger, anticipation, disgust, fear, joy, sadness, surprise, trust); all tied candidates and scores remain visible. Zero total score yields `NONE`, not an arbitrary emotion. The LLM uses the same eight emotional categories but is forced to choose the closest one; this difference in abstention policy is documented.

## Repeat inference deliberately

The submitted output is already complete. `run_reviews.py` resumes completed source IDs and will not repeat saved reviews. Set credentials in your shell, not in project files:

```bash
export OPENAI_BASE_URL='http://dobolyi.com:9001/v1'
export OPENAI_API_KEY='your-course-key'
export OPENAI_MODEL='cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit'
python run_reviews.py
python add_emotions.py
python build_dashboard.py
python write_report.py
```

The HTTP endpoint is unencrypted; do not send sensitive private data. The code checks `/models` rather than assuming it hosts an OpenAI model. Sampling is reproducible, but remote inference is not guaranteed identical even at temperature zero. For a fresh run, work in a separate copy and remove that copy's `final_raw.jsonl`, `results_joint.json`, and `results_final.json` deliberately before invoking the runner. Never delete the submission evidence in place.

`classify.py` contains the shared sampling and scoring helpers plus the earlier sentiment-only runner; the final combined entry point is **`run_reviews.py`**. Historical binary evidence is in `evidence/`. The original CSV is retained exactly (including its legacy quoting convention) rather than represented as a clean modern export.

## Automated checks and screenshots

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python -m unittest test_three_class test_emotions test_integrity -v
python verify_submission.py
python test_dashboard.py
```

Browser tests use offline mode, compare rendered metrics/matrices/counts to saved output, exercise review filters and empty states, and measure chart element widths at desktop and mobile sizes. Screenshots are written under `screenshots/`. Inspect the screenshots yourself as well as reading the test results.

`python write_report.py` regenerates the **agent draft** from result files and `SOURCES.md`. Do not run it over your own final edits without backing them up. Your personal check, rewriting of conclusions, and submission remain required.
