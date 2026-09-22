"""Final joint sentiment/emotion run over the fixed balanced sample; resumable."""
if not __debug__:
    raise SystemExit('Integrity checks require assertions: run without -O/-OO and unset PYTHONOPTIMIZE.')

import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI
from classify import LABELS, DATA, sample_balanced, score

ROOT=Path(__file__).resolve().parent
EMOTIONS=('anger','anticipation','disgust','fear','joy','sadness','surprise','trust')

def parse_output(content,finish):
    if finish!='stop' or not content: return 'UNKNOWN','UNKNOWN'
    try:
        obj=json.loads(content)
        if not isinstance(obj,dict) or set(obj)!={'sentiment','emotion'}: return 'UNKNOWN','UNKNOWN'
        if obj['sentiment'] not in LABELS or obj['emotion'] not in EMOTIONS: return 'UNKNOWN','UNKNOWN'
        return obj['sentiment'],obj['emotion']
    except (ValueError,TypeError): return 'UNKNOWN','UNKNOWN'

def main():
    prompt=(ROOT/'prompt.txt').read_text().strip()
    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
    manifest=json.loads((ROOT/'balanced_sample.json').read_text())
    rows=manifest['reviews']
    sample,meta=sample_balanced(DATA,50,6418)
    assert sample==rows and len(rows)==150
    assert manifest['metadata']['source_sha256']==hashlib.sha256(DATA.read_bytes()).hexdigest()
    key=os.environ.get('OPENAI_API_KEY')
    if not key: raise SystemExit('Set OPENAI_API_KEY in your environment')
    client=OpenAI(base_url=os.environ.get('OPENAI_BASE_URL','http://dobolyi.com:9001/v1'),api_key=key,timeout=180,max_retries=2)
    models=[x.id for x in client.models.list().data]
    model=os.environ.get('OPENAI_MODEL') or (models[0] if len(models)==1 else None)
    if model not in models: raise SystemExit('Set OPENAI_MODEL to an available model ID')
    raw=ROOT/'final_raw.jsonl'
    existing=[json.loads(l) for l in raw.read_text().splitlines()] if raw.exists() else []
    indexed={r['id']:r for r in rows}
    assert len({r['id'] for r in existing})==len(existing)
    for r in existing:
        assert all(r[k]==indexed[r['id']][k] for k in indexed[r['id']])
        assert r['prompt_sha256']==prompt_hash and r['model']==model
    done={r['id'] for r in existing}
    with raw.open('a',encoding='utf-8') as f:
        for review in rows:
            if review['id'] in done: continue
            attempts=[]; pred=emotion='UNKNOWN'
            for budget in (4096,8192):
                try:
                    response=client.chat.completions.create(model=model,messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps({'title':review['title'],'text':review['text']},ensure_ascii=False)}],temperature=0,max_tokens=budget)
                    choice=response.choices[0]
                    pred,emotion=parse_output(choice.message.content,choice.finish_reason)
                    # Raw final content is saved verbatim. Reasoning is deliberately not parsed or published.
                    attempts.append(dict(response_id=response.id,final_content=choice.message.content,finish_reason=choice.finish_reason,max_tokens=budget,usage=response.usage.model_dump() if response.usage else None))
                except Exception as e:
                    attempts.append(dict(error_type=type(e).__name__,max_tokens=budget))
                if pred!='UNKNOWN': break
            r=dict(review,pred=pred,correct=pred==review['truth'],llm_emotion=emotion,model=model,prompt_sha256=prompt_hash,completed_at=datetime.now(timezone.utc).isoformat(),attempts=attempts)
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();existing.append(r)
            print(f"{len(existing)}/150 row={r['id']} {r['pred']} / {r['llm_emotion']}",flush=True)
    existing.sort(key=lambda r:r['id'])
    metadata=dict(manifest['metadata'],system_prompt=prompt,prompt_sha256=prompt_hash)
    result=dict(metadata=metadata,model=model,metrics=score(existing),reviews=existing)
    (ROOT/'results_joint.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result['metrics'],indent=2))

if __name__=='__main__': main()
