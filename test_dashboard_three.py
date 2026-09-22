"""Offline final-report acceptance tests. Run after build_dashboard.py; --static needs no data."""
if not __debug__:
    raise SystemExit('Integrity checks require assertions: run without -O/-OO and unset PYTHONOPTIMIZE.')

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

def static_checks():
    template = (ROOT/'dashboard_template.html').read_text()
    builder = (ROOT/'build_dashboard.py').read_text()
    assert "results_final.json" in builder, 'Builder must consume final joint report'
    assert "final_raw.jsonl" in builder, 'Hash the final raw predictions'
    for identifier in ['emotions', 'emotion-agreement', 'emotion-coverage', 'emotion-bars', 'emotion-result', 'llm-emotion', 'nrc-emotion']:
        assert f'id="{identifier}"' in template, identifier
    assert 'no primary-emotion analysis' not in template


def pct(value):
    return f'{value*100:.1f}'.removesuffix('.0')+'%'


def main():
    static_checks()
    if '--static' in sys.argv:
        print('PASS: final-report template/builder contract')
        return
    from playwright.sync_api import sync_playwright
    report = json.loads((ROOT/'results_final.json').read_text())
    rows, m, em = report['reviews'], report['metrics'], report['emotion_metrics']
    total = len(rows)
    cols = LABELS + (['UNKNOWN'] if m['unknown'] else [])
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(offline=True, viewport={'width':1360,'height':1000})
        page = context.new_page()
        errors, requests = [], []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: requests.append(r.url) if r.url.startswith(('http:', 'https:')) else None)
        page.goto((ROOT/'results_dashboard.html').as_uri())
        # Provenance lives inside a collapsed details element; open it before testing visible text.
        page.locator('.method details').evaluate('e=>e.open=true')
        assert page.locator('#data').evaluate('e=>JSON.parse(e.textContent).reviews.map(r=>r.id)') == [r['id'] for r in rows]
        assert page.locator('#prediction-hash').inner_text() == hashlib.sha256((ROOT/'final_raw.jsonl').read_bytes()).hexdigest()
        for selector, expected in {'run-size':str(total), 'accuracy':pct(m['accuracy']), 'neutral-recall':pct(m['per_class']['NEUTRAL']['recall']), 'macro-f1':pct(m['macro_f1'])}.items():
            assert page.locator('#'+selector).inner_text() == expected, selector
        assert page.locator('#correct').inner_text() == f"{m['correct']} / {total} correct · {total-m['correct']} mismatches"
        neutral = m['per_class']['NEUTRAL']
        assert page.locator('#neutral-correct').inner_text() == f"{neutral['correct']} / {neutral['support']} three-star reviews recovered"
        assert page.locator('#baseline').inner_text() == f"Single-class baseline: {pct(m['baseline'])}. Balanced accuracy: {pct(m['balanced_accuracy'])} (same as overall accuracy because supports are equal)."
        assert page.locator('#unknowns').inner_text() == f"{m['unknown']} unknown outputs. Denominators include all {total} reviews, including any failed predictions."
        assert page.locator('#neutral-destinations').locator('span').all_inner_texts() == [f"{m['matrix']['NEUTRAL'][c]} {c.capitalize()} · {pct(m['matrix']['NEUTRAL'][c]/neutral['support'])}" for c in cols]
        assert page.locator('#lexicon-hash').text_content() == report['metadata']['lexicon_sha256']
        for i, label in enumerate(LABELS):
            v = m['per_class'][label]
            assert page.locator('#scorecard tr').nth(i).locator('td').all_inner_texts() == [str(v['support']),str(v['predicted']),pct(v['precision']),pct(v['recall']),pct(v['f1'])]
            assert page.locator('#recall-bars .bar-group').nth(i).locator('.rowhead b').last.inner_text() == pct(v['recall'])
        cells = page.locator('#matrix td[data-count]')
        assert cells.evaluate_all('els=>els.map(e=>Number(e.dataset.count))') == [m['matrix'][t][c] for t in LABELS for c in cols]
        for i, (t,c) in enumerate(itertools.product(LABELS,cols)):
            assert cells.nth(i).locator('strong').inner_text() == str(m['matrix'][t][c])
            assert cells.nth(i).locator('small').inner_text() == pct(m['matrix'][t][c]/m['per_class'][t]['support'])+' of row'
        assert page.locator('#emotion-agreement').inner_text() == pct(em['agreement'])
        assert page.locator('#emotion-all-detail').inner_text() == f"{em['agree']} / {total} reviews · all outputs included"
        assert page.locator('#emotion-coverage').inner_text() == (pct(em['covered_agree']/em['covered']) if em['covered'] else 'N/A')
        assert page.locator('#emotion-covered-detail').inner_text() == f"{em['covered_agree']} / {em['covered']} covered reviews"
        for field in ['no_hits','tied','unique_top','unique_top_agree','covered']:
            assert page.locator(f'[data-emetric="{field}"]').inner_text() == str(em[field]), field
        assert page.locator('#emotion-unique').inner_text() == (pct(em['unique_top_agree']/em['unique_top']) if em['unique_top'] else 'N/A')
        emotion_labels = EMOTIONS + ['NONE', 'UNKNOWN']
        assert page.locator('#emotion-bars [data-count]').evaluate_all('els=>els.map(e=>Number(e.dataset.count))') == [em[key].get(e,0) for e in emotion_labels for key in ['llm_counts','nrc_counts']]
        assert page.locator('.cell').evaluate_all('els=>els.map(e=>Number(e.dataset.id))') == [r['id'] for r in rows]

        def check_ids(expected):
            assert page.locator('.review').evaluate_all('els=>els.map(e=>Number(e.dataset.id))') == [r['id'] for r in expected]
            assert page.locator('#count').inner_text().startswith(f'{len(expected)} of {total}')

        for outcome, truth, pred in itertools.product(['all','right','wrong'], ['all']+LABELS, ['all']+LABELS+['UNKNOWN']):
            for key,value in [('result',outcome),('cls',truth),('pred',pred)]:
                page.locator('#'+key).select_option(value)
            check_ids([r for r in rows if (outcome=='all' or r['correct']==(outcome=='right')) and (truth=='all' or r['truth']==truth) and (pred=='all' or r['pred']==pred)])
        page.locator('#reset').click()
        for outcome, llm, nrc in itertools.product(['all','agree','disagree','no-hit','tied','unique'], ['all']+EMOTIONS+['UNKNOWN'], ['all']+EMOTIONS+['NONE']):
            for key,value in [('emotion-result',outcome),('llm-emotion',llm),('nrc-emotion',nrc)]:
                page.locator('#'+key).select_option(value)
            def matches(r):
                status = {'all':True,'agree':r['emotion_agrees'],'disagree':not r['emotion_agrees'],'no-hit':r['nrc']['emotion']=='NONE','tied':len(r['nrc']['ties'])>1,'unique':len(r['nrc']['ties'])==1}[outcome]
                return status and (llm=='all' or r['llm_emotion']==llm) and (nrc=='all' or r['nrc']['emotion']==nrc)
            check_ids([r for r in rows if matches(r)])
        page.locator('#reset').click()
        # Verify full text and every score from rendered review details, not the embedded report.
        for r in rows:
            review = page.locator(f'#review-{r["id"]}')
            assert review.locator('.review-text').text_content() == (r['text'] or '(No body text)')
            assert review.locator('.llm-label').inner_text() == r['llm_emotion']
            assert review.locator('.nrc-label').inner_text() == r['nrc']['emotion']
            assert review.locator('[data-emotion-score]').evaluate_all('els=>Object.fromEntries(els.map(e=>[e.dataset.emotionScore,Number(e.textContent)]))') == r['nrc']['scores']
        page.locator('#search').fill('zz-no-match-zz')
        check_ids([])
        page.locator('#reset').click()
        for t,c in itertools.product(LABELS,cols):
            page.locator(f'#matrix button[data-truth="{t}"][data-pred="{c}"]').click()
            check_ids([r for r in rows if r['truth']==t and r['pred']==c])
        first = rows[0]['id']
        page.locator(f'.cell[data-id="{first}"]').click()
        check_ids([rows[0]])
        assert page.locator(f'#review-{first} details').get_attribute('open') is not None
        page.locator('#neutral-focus').click()
        check_ids([r for r in rows if r['truth']=='NEUTRAL' and not r['correct']])
        page.locator('#emotion-focus').click()
        check_ids([r for r in rows if not r['emotion_agrees']])
        # AND across sentiment and emotion filters, not just within each group.
        page.locator('#cls').select_option('NEUTRAL')
        page.locator('#result').select_option('wrong')
        check_ids([r for r in rows if not r['emotion_agrees'] and r['truth']=='NEUTRAL' and not r['correct']])
        page.locator('#reset').click()

        def geometry(selector, denominator):
            bars = page.locator(selector).evaluate_all('els=>els.map(e=>({count:Number(e.dataset.count),width:e.querySelector(".fill").getBoundingClientRect().width,track:e.querySelector(".track").getBoundingClientRect().width,text:e.textContent}))')
            for bar in bars:
                count,width,track = bar['count'],bar['width'],bar['track']
                expected = max(2,track*count/denominator) if count else 0
                assert math.isclose(width,expected,abs_tol=.1), (selector,bar,expected)
                assert (width>=1.9 if count else width==0), bar
                assert str(count) in bar['text'].replace(',',''), bar
        for width,height in [(1360,1000),(390,844)]:
            page.set_viewport_size({'width':width,'height':height})
            for source in ['source','sample']:
                page.locator('#star-source').select_option(source)
                expected = [report['metadata']['source_stars'][str(i)] if source=='source' else sum(r['rating']==i for r in rows) for i in range(1,6)]
                assert page.locator('#stars [data-count]').evaluate_all('els=>els.map(e=>Number(e.dataset.count))') == expected
                geometry('#stars [data-count]',report['metadata']['source_rows'] if source=='source' else total)
            assert page.locator('#prediction-bars [data-count]').evaluate_all('els=>els.map(e=>Number(e.dataset.count))') == [m['per_class'].get(c,{'support':0,'predicted':m['unknown']})[k] for c in cols for k in ['support','predicted']]
            geometry('#prediction-bars [data-count]',total)
            geometry('#emotion-bars [data-count]',total)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), width
        for theme in ['ink','sage']:
            page.locator('#theme').select_option(theme)
        assert not errors, errors
        assert not requests, requests
        # Only capture actual final data, after every acceptance assertion passes.
        shots = ROOT/'screenshots'
        shots.mkdir(exist_ok=True)
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(shots/'mobile.png'))
        page.set_viewport_size({'width':1360,'height':1000})
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(shots/'dashboard.png'))
        page.locator('#emotions').evaluate('e=>e.scrollIntoView({block:"start"})')
        page.screenshot(path=str(shots/'emotions.png'))
        browser.close()
        verification = dict(status='passed', source='results_final.json',
            sentiment_filter_combinations=len(list(itertools.product(['all','right','wrong'],['all']+LABELS,['all']+LABELS+['UNKNOWN']))),
            emotion_filter_combinations=len(list(itertools.product(['all','agree','disagree','no-hit','tied','unique'],['all']+EMOTIONS+['UNKNOWN'],['all']+EMOTIONS+['NONE']))),
            records=total, sentiment_dom='passed', emotion_dom='passed',
            exact_review_ids='passed', all_eight_nrc_scores='passed',
            desktop_mobile_bar_geometry='passed', unexpected_network_requests=len(requests), javascript_errors=len(errors),
            screenshots=['screenshots/dashboard.png','screenshots/emotions.png','screenshots/mobile.png'])
        (ROOT/'evidence').mkdir(exist_ok=True)
        (ROOT/'evidence/browser_verification.json').write_text(json.dumps(verification,indent=2))
        print('PASS: final sentiment/emotion DOM metrics, 60 sentiment + 600 emotion filter combinations, exact IDs/full text/eight scores, matrix/tile, desktop/mobile bar geometry, offline/no errors; three viewport screenshots saved')

if __name__ == '__main__':
    main()
