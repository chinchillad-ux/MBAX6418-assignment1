"""Independent evidence validation; no API calls or network."""
if not __debug__:
    raise SystemExit('Integrity checks require assertions: run without -O/-OO and unset PYTHONOPTIMIZE.')

import json
import hashlib
from pathlib import Path
from collections import Counter
from classify import LABEL_FROM_RATING, sample_balanced, score, DATA
from run_reviews import parse_output
from add_emotions import load_lexicon, word_emotion

ROOT=Path(__file__).resolve().parent

def main():
    r=json.loads((ROOT/'results_final.json').read_text())
    raw=[json.loads(line) for line in (ROOT/'final_raw.jsonl').read_text().splitlines()]
    manifest=json.loads((ROOT/'balanced_sample.json').read_text())
    assert len(raw)==len(r['reviews'])==150
    assert len({x['id'] for x in raw})==150
    sample,_=sample_balanced(DATA,50,6418)
    assert sample==manifest['reviews']
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==r['metadata']['source_sha256']
    prompt=(ROOT/'prompt.txt').read_text().strip()
    assert prompt==r['metadata']['system_prompt']
    ph=hashlib.sha256(prompt.encode()).hexdigest()
    lookup={x['id']:x for x in raw}
    lex_path=ROOT/'resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    assert hashlib.sha256(lex_path.read_bytes()).hexdigest()==r['metadata']['lexicon_sha256']
    lex=load_lexicon(lex_path)
    for record,source in zip(r['reviews'],sample):
        assert all(record[k]==source[k] for k in source)
        a=lookup[record['id']]
        assert all(record[k]==a[k] for k in a)
        last=a['attempts'][-1]
        assert parse_output(last.get('final_content'),last.get('finish_reason'))==(a['pred'],a['llm_emotion'])
        assert a['prompt_sha256']==ph
        assert a['truth']==LABEL_FROM_RATING(a['rating'])
        assert record['nrc']==word_emotion(a['title'],a['text'],lex)
        assert record['emotion_agrees']==(a['llm_emotion']==record['nrc']['emotion'])
    assert r['metrics']==score(raw)
    rows=r['reviews']
    covered=[x for x in rows if x['nrc']['emotion']!='NONE' and x['llm_emotion']!='UNKNOWN']
    unique=[x for x in covered if len(x['nrc']['ties'])==1]
    agree=sum(x['emotion_agrees'] for x in rows)
    expected=dict(total=len(rows),agree=agree,agreement=agree/len(rows),
        covered=len(covered),covered_agree=sum(x['emotion_agrees'] for x in covered),
        no_hits=sum(x['nrc']['emotion']=='NONE' for x in rows),
        tied=sum(len(x['nrc']['ties'])>1 for x in rows),
        unique_top=len(unique),unique_top_agree=sum(x['emotion_agrees'] for x in unique),
        llm_counts=dict(Counter(x['llm_emotion'] for x in rows)),
        nrc_counts=dict(Counter(x['nrc']['emotion'] for x in rows)),
        transitions=dict(Counter(x['llm_emotion']+' → '+x['nrc']['emotion'] for x in rows)))
    assert r['emotion_metrics']==expected, 'Emotion metrics mismatch'
    result=dict(status='passed',records=150,unique_ids=150,source_identity='passed',raw_final_answers='passed',sentiment_metrics='passed',nrc_scores='passed',emotion_metrics='passed')
    (ROOT/'evidence/data_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
