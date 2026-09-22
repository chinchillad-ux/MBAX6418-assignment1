"""Regenerate the agent-drafted report from saved result files, not hand-entered metrics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results_final.json').read_text());m=r['metrics'];e=r['emotion_metrics'];meta=r['metadata']
old=json.loads((ROOT/'evidence/legacy_binary_summary.json').read_text())
pct=lambda x:f'{x*100:.1f}%'
matrix='\n'.join('| '+c.title()+' | '+' | '.join(str(m['matrix'][c][p]) for p in ('POSITIVE','NEUTRAL','NEGATIVE'))+' | '+pct(m['per_class'][c]['recall'])+' |' for c in ('POSITIVE','NEUTRAL','NEGATIVE'))
text=f'''# Gift-card reviews: sentiment and emotion audit

> **Agent-drafted report; student review is still required.** The agent generated the code, pulled these numbers from saved output, and drafted this narrative. Before submitting, check the figures yourself and revise the interpretations into your own words. Automated checks do not replace that personal review.

![Final sentiment dashboard](screenshots/dashboard.png)

## Data and method

Data: **Amazon Reviews ’23**, McAuley Lab, **Gift Cards** category, from the dataset page and its linked review download.[1] The supplied file contains **{meta['source_rows']:,} reviews**, verified by scanning it. The selected manifest is [balanced_sample.json](balanced_sample.json).

The final run samples **{meta['per_class']} reviews per class ({m['total']} total)** from the entire file without replacement, using fixed seed **{meta['seed']}**. Labels are **4–5 stars = POSITIVE, 3 = NEUTRAL, 1–2 = NEGATIVE**. Only title and text go to `{r['model']}`, served through the course’s OpenAI-compatible endpoint; this is not an OpenAI model. [The prompt](prompt.txt) requests sentiment and one primary emotion together. The model does not see the rating.

## Why the lopsided run looked so good

The original binary run matched **{old['correct']}/{old['total']} ({pct(old['accuracy'])})**, but its labels were **{old['class_counts']['POSITIVE']} positive and {old['class_counts']['NEGATIVE']} negative**. Always predicting positive already scored **{pct(old['baseline'])}**. Three-star reviews were folded into negative, so neutral-class failure was invisible. See [the saved legacy summary](evidence/legacy_binary_summary.json).

The final balanced run scores **{m['correct']}/{m['total']} ({pct(m['accuracy'])})**, versus a **{pct(m['baseline'])}** single-class baseline. Macro F1 is **{pct(m['macro_f1'])}**; unknown outputs: **{m['unknown']}**. Balanced accuracy equals overall accuracy because class sizes are equal. These results describe the balanced test, not the source file’s natural class prevalence. This is not a controlled before/after comparison: sample, class definitions, prompt, and parsing changed. The final joint prompt also differs from the earlier sentiment-only balanced run; all figures here use the final joint output.

## Where the mistakes go

Rows are actual rating labels; columns are model predictions. Source: [results_final.json](results_final.json), `metrics.matrix`.

| Actual | Positive | Neutral | Negative | Recall |
|---|---:|---:|---:|---:|
{matrix}

The main direction is **neutral → negative ({m['matrix']['NEUTRAL']['NEGATIVE']})**, followed by **neutral → positive ({m['matrix']['NEUTRAL']['POSITIVE']})**. Only **{m['matrix']['NEUTRAL']['NEUTRAL']}/{meta['per_class']}** three-star reviews remain neutral; the reverse negative → neutral error occurs **{m['matrix']['NEGATIVE']['NEUTRAL']}** time. Neutral-rated reviews account for **{meta['per_class']-m['matrix']['NEUTRAL']['NEUTRAL']} of {m['total']-m['correct']} mismatches**. Balancing exposes a weak middle class that strong performance at the extremes had hidden. A mismatch is against a rating proxy, not proof that the model misread the text: a three-star review can still contain clearly favorable or critical language.

## LLM versus word-list emotions

![Emotion comparison](screenshots/emotions.png)

The independent word-list method uses the **NRC Emotion Lexicon (EmoLex)** by **Saif M. Mohammad and Peter D. Turney, National Research Council Canada**.[2][3][4]

The associated papers are *Crowdsourcing a Word-Emotion Association Lexicon* (2013) and *Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon* (2010).[3][4]

The word-list script lowercases and tokenizes title plus text, counts repeated words, sums the eight emotion associations, and takes the largest score. Ties select the first alphabetical emotion while retaining all tied candidates; no matches produce `NONE`. It does not handle negation, inflections, context, or word sense. The LLM is instead asked to choose the closest contextual emotion from those eight, even when emotional evidence is weak.

The methods agree on **{e['agree']}/{e['total']} ({pct(e['agreement'])})**. NRC has **{e['no_hits']} no-hit reviews** and **{e['tied']} tied maxima**. Among covered reviews agreement is **{e['covered_agree']}/{e['covered']} ({pct(e['covered_agree']/e['covered']) if e['covered'] else 'N/A'})**; among unique-maximum reviews it is **{e['unique_top_agree']}/{e['unique_top']} ({pct(e['unique_top_agree']/e['unique_top']) if e['unique_top'] else 'N/A'})**. These are agreement rates, **not emotion accuracy**: there are no human emotion labels.

The LLM most often assigns **anger ({e['llm_counts'].get('anger',0)})** and **joy ({e['llm_counts'].get('joy',0)})**; NRC most often assigns **anticipation ({e['nrc_counts'].get('anticipation',0)})**. The tie rule contributes to that skew, not just the review content. In source row **4444**, “Good gift,” the LLM chooses joy, while NRC ties anticipation, joy, and surprise and selects anticipation. In row **7443**, “so cute,” the LLM chooses joy but the exact-match lexicon has no emotion-bearing tokens. Those examples illustrate tie-breaking and vocabulary coverage—not evidence that one method is universally better. Full text, both labels, and all NRC scores are inspectable in the dashboard.

## Bugs, process issues, and checks

- The initial macOS `zcat` command failed; Python’s `gzip` reader replaced it. An assumed `review_id` field was absent; stable source row numbers now identify reviews.
- The early confusion matrix mixed short and full label names and crashed after inference. Existing predictions were rescored rather than requested again.
- The original parser could extract a label from unfinished reasoning, and the original CSV writer used JSON escaping instead of CSV quoting. The final runner accepts only schema-valid, completed final JSON; raw final content is retained verbatim. Structured JSONL is now the canonical output.
- An earlier narrative over-attributed a mistake to its title. Reading the full body revealed a real complaint; the report now separates observed outcomes from possible explanations.
- A tool timeout interrupted the joint run. Incrementally saved records allowed resumption by source ID without repeating completed reviews.
- A clean-checkout test caught the NRC server rejecting Python’s default user agent. The downloader now uses a tested accepted request header; a fresh checkout downloads the original resources and reproduces the report without new model calls.
- Browser checks compare DOM figures with saved output, test filters and live counts, and measure **nonzero bar widths** at desktop and mobile sizes. Small bars are checked geometrically, not just by eye. No collapsed bars were observed in the final verification. Screenshots show the tested interface.

## Working files and reproduction

| Deliverable | File |
|---|---|
| Exact final prompt | [prompt.txt](prompt.txt) |
| Review runner + scoring helpers | [run_reviews.py](run_reviews.py), [classify.py](classify.py) |
| Independent NRC scorer | [add_emotions.py](add_emotions.py) |
| Dashboard generator + template | [build_dashboard.py](build_dashboard.py), [dashboard_template.html](dashboard_template.html) |
| Balanced raw model output | [final_raw.jsonl](final_raw.jsonl) — verbatim final answers, status and request metadata; reasoning excluded |
| Scored sentiment + emotion output | [results_final.json](results_final.json) |
| Final offline dashboard | [results_dashboard.html](results_dashboard.html) — download and open locally |
| Reproduction and browser checks | [SETUP.md](SETUP.md), [test_dashboard_three.py](test_dashboard_three.py) |

The dashboard contains all review text and works offline. The lexicon is **not redistributed**; its terms prohibit redistribution, and SETUP explains how to obtain it from the author.[3] The public repository belongs to the submitting student’s account and is readable by anyone with its link. The instructor’s GitHub account, `davedgd`, is also confirmed as a collaborator.

### Student sign-off — complete before handing in the link

- [ ] I checked the README numbers against the saved results and dashboard.
- [ ] I read the example reviews and revised the conclusions into my own words.
- [ ] I confirmed the instructor can access my own repository and submitted its link.

'''
(ROOT/'README.md').write_text(text+(ROOT/'SOURCES.md').read_text())
print('README drafted from saved final and legacy metrics, with registered sources.')
