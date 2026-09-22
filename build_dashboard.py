"""Validate and render final sentiment + NRC report. No network/API calls."""
if not __debug__:
    raise SystemExit('Integrity checks require assertions: run without -O/-OO and unset PYTHONOPTIMIZE.')

import hashlib
import json
from collections import Counter
from pathlib import Path
from classify import sample_balanced, score, DATA, LABELS
from add_emotions import EMOTIONS, load_lexicon, word_emotion
from run_reviews import parse_output

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT/'results_final.json').read_text())
rows = report['reviews']
selected, metadata = sample_balanced(DATA, report['metadata']['per_class'], report['metadata']['seed'])
manifest = json.loads((ROOT/'balanced_sample.json').read_text())
assert len(rows) == len(selected) == 150
assert len({r['id'] for r in rows}) == 150
assert manifest['reviews'] == selected
assert report['metadata']['source_sha256'] == hashlib.sha256(DATA.read_bytes()).hexdigest()
# Joint prompt intentionally differs from the sentiment-only sample manifest.
for key in metadata:
    assert report['metadata'][key] == metadata[key], key
raw_path = ROOT/'final_raw.jsonl'
raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
assert len(raw) == len(rows) and len({r['id'] for r in raw}) == len(rows)
raw_by_id = {r['id']: r for r in raw}
lex_path = ROOT/'resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
assert report['metadata']['lexicon_sha256'] == hashlib.sha256(lex_path.read_bytes()).hexdigest()
lexicon = load_lexicon(lex_path)
for r, s in zip(rows, selected):
    assert all(r[k] == s[k] for k in s), f"Source mismatch: {r['id']}"
    assert r['correct'] == (r['truth'] == r['pred'])
    assert r['model'] == report['model']
    assert r['prompt_sha256'] == report['metadata']['prompt_sha256'] == hashlib.sha256(report['metadata']['system_prompt'].encode()).hexdigest()
    last = r['attempts'][-1]
    assert (r['pred'], r['llm_emotion']) == parse_output(last.get('final_content'), last.get('finish_reason'))
    saved = raw_by_id[r['id']]
    assert all(r[k] == value for k, value in saved.items()), f"Raw prediction mismatch: {r['id']}"
    assert r['nrc'] == word_emotion(r['title'], r['text'], lexicon)
    assert r['llm_emotion'] in (*EMOTIONS, 'UNKNOWN')
    assert r['emotion_agrees'] == (r['llm_emotion'] == r['nrc']['emotion'])
assert Counter(r['truth'] for r in rows) == {c:50 for c in LABELS}
assert report['metrics'] == score(rows)
covered = [r for r in rows if r['nrc']['emotion'] != 'NONE' and r['llm_emotion'] != 'UNKNOWN']
unique = [r for r in covered if len(r['nrc']['ties']) == 1]
agree = sum(r['emotion_agrees'] for r in rows)
expected = dict(total=len(rows), agree=agree, agreement=agree/len(rows),
    covered=len(covered), covered_agree=sum(r['emotion_agrees'] for r in covered),
    no_hits=sum(r['nrc']['emotion']=='NONE' for r in rows),
    tied=sum(len(r['nrc']['ties'])>1 for r in rows), unique_top=len(unique),
    unique_top_agree=sum(r['emotion_agrees'] for r in unique),
    llm_counts=dict(Counter(r['llm_emotion'] for r in rows)),
    nrc_counts=dict(Counter(r['nrc']['emotion'] for r in rows)),
    transitions=dict(Counter(r['llm_emotion']+' → '+r['nrc']['emotion'] for r in rows)))
assert report['emotion_metrics'] == expected
report['metadata']['prediction_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
for r in rows:
    r['final_content'] = r['attempts'][-1].get('final_content') or ''
    r['finish_reason'] = r['attempts'][-1].get('finish_reason') or 'error'
    del r['attempts']
payload = json.dumps(report, ensure_ascii=False).replace('<', '\\u003c')
html = (ROOT/'dashboard_template.html').read_text().replace('__REPORT__', payload)
(ROOT/'results_dashboard.html').write_text(html)
print('Validated 150 source identities, raw predictions, sentiment scores, NRC word scores and emotion metrics; wrote results_dashboard.html')
