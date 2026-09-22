"""Add NRC word-list emotions to saved LLM output. No model calls."""
import argparse
import collections
import hashlib
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EMOTIONS=('anger','anticipation','disgust','fear','joy','sadness','surprise','trust')

def load_lexicon(path):
    lex={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields=line.split('\t')
        if len(fields)!=3: continue
        word,emotion,flag=fields
        if emotion in EMOTIONS and flag=='1': lex.setdefault(word,{})[emotion]=1
    if not lex: raise ValueError('No emotion associations found in NRC word-level file')
    return lex

def word_emotion(title,text,lex):
    text=html.unescape(title+' '+text).lower()
    text=re.sub(r'<[^>]*>',' ',text)
    tokens=re.findall(r"[a-z]+(?:'[a-z]+)?",text)
    counts=collections.Counter(tokens)
    scores={e:0 for e in EMOTIONS}
    matched={}
    for word,count in counts.items():
        if word in lex:
            matched[word]=count
            for e in EMOTIONS: scores[e]+=count*lex[word].get(e,0)
    top=max(scores.values())
    ties=[e for e in EMOTIONS if scores[e]==top] if top else []
    return dict(emotion=ties[0] if ties else 'NONE',scores=scores,ties=ties,
                matched_tokens=matched,token_count=len(tokens))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lexicon',type=Path,default=ROOT/'resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    args=ap.parse_args()
    if not args.lexicon.exists(): raise SystemExit('Download the NRC lexicon from its author page; see SETUP.md. It is not redistributed.')
    lex=load_lexicon(args.lexicon)
    report=json.loads((ROOT/'results_joint.json').read_text())
    for r in report['reviews']:
        result=word_emotion(r['title'],r['text'],lex)
        r['nrc']=result
        r['emotion_agrees']=r['llm_emotion']==result['emotion']
    rows=report['reviews']; total=len(rows)
    covered=[r for r in rows if r['nrc']['emotion']!='NONE' and r['llm_emotion']!='UNKNOWN']
    untied=[r for r in covered if len(r['nrc']['ties'])==1]
    agree=sum(r['emotion_agrees'] for r in rows)
    report['emotion_metrics']=dict(total=total,agree=agree,agreement=agree/total,
        covered=len(covered),covered_agree=sum(r['emotion_agrees'] for r in covered),
        no_hits=sum(r['nrc']['emotion']=='NONE' for r in rows),
        tied=sum(len(r['nrc']['ties'])>1 for r in rows),
        unique_top=len(untied),unique_top_agree=sum(r['emotion_agrees'] for r in untied),
        llm_counts=dict(collections.Counter(r['llm_emotion'] for r in rows)),
        nrc_counts=dict(collections.Counter(r['nrc']['emotion'] for r in rows)),
        transitions=dict(collections.Counter(r['llm_emotion']+' → '+r['nrc']['emotion'] for r in rows)))
    report['metadata']['lexicon_sha256']=hashlib.sha256(args.lexicon.read_bytes()).hexdigest()
    report['metadata']['lexicon_source']='https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm'
    report['metadata']['lexicon_version']='NRC Emotion Lexicon word-level v0.92'
    report['metadata']['lexicon_policy']='Count repeated lowercase English word tokens in title+text; no stemming or negation handling; sum eight emotion flags; select first alphabetical maximum; NONE when all scores are zero. Preserve all tied maxima.'
    (ROOT/'results_final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['emotion_metrics'],indent=2))

if __name__=='__main__': main()
