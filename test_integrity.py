"""Offline robustness regressions; all generated fixtures stay in scratch."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SCRATCH = Path(tempfile.gettempdir())  # Honors TMPDIR; portable across checkouts.


class IntegrityTests(unittest.TestCase):
    def run_script(self, root, name):
        # Prevent accidental network/model access in every fixture subprocess.
        bootstrap = '''
import runpy, sys
def offline(event, args):
    if event.startswith('socket.') or event == 'subprocess.Popen':
        raise RuntimeError('Network/process access forbidden in offline fixture')
sys.addaudithook(offline)
target = sys.argv[1]
sys.argv = [target]
runpy.run_path(target, run_name='__main__')
'''
        env = dict(os.environ)
        env.pop('PYTHONOPTIMIZE', None)
        return subprocess.run([sys.executable, '-B', '-c', bootstrap, name],
                              cwd=root, env=env, text=True, capture_output=True, timeout=30)

    def make_fixture(self, root, title='happy'):
        """Explicit synthetic input, never altered or substituted delivered results."""
        from classify import sample_balanced, score
        for name in ('classify.py', 'run_reviews.py', 'add_emotions.py',
                     'verify_submission.py', 'write_report.py'):
            shutil.copy2(ROOT / name, root / name)
        (root / 'resources').mkdir()
        (root / 'evidence').mkdir()
        source = root / 'Gift_Cards.jsonl.gz'
        with gzip.open(source, 'wt') as out:
            for rating in (1, 3, 5):
                for _ in range(50):
                    out.write(json.dumps(dict(rating=rating, title=title, text='fixture')) + '\n')
        rows, metadata = sample_balanced(source, 50, 6418)
        prompt = 'Explicit offline regression fixture; no inference.'
        ph = hashlib.sha256(prompt.encode()).hexdigest()
        metadata.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        system_prompt=prompt, prompt_sha256=ph)
        (root / 'prompt.txt').write_text(prompt)
        (root / 'balanced_sample.json').write_text(json.dumps(dict(metadata=metadata, reviews=rows)))
        raw = [dict(row, pred=row['truth'], correct=True, llm_emotion='joy',
                    model='offline-fixture', prompt_sha256=ph,
                    attempts=[dict(final_content=json.dumps(dict(sentiment=row['truth'], emotion='joy')),
                                   finish_reason='stop')]) for row in rows]
        (root / 'final_raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in raw))
        (root / 'results_joint.json').write_text(json.dumps(dict(metadata=metadata,
            model='offline-fixture', metrics=score(raw), reviews=raw)))
        (root / 'resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt').write_text(
            'happy\tjoy\t1\ntied\tanger\t1\ntied\tjoy\t1\n')
        generated = self.run_script(root, 'add_emotions.py')
        self.assertEqual(generated.returncode, 0, generated.stderr)
        return json.loads((root / 'results_final.json').read_text())

    def test_complete_emotion_metrics_tampering_rejected(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            original = self.make_fixture(root)
            valid = self.run_script(root, 'verify_submission.py')
            self.assertEqual(valid.returncode, 0, valid.stderr)
            evidence = root / 'evidence/data_verification.json'
            self.assertEqual(json.loads(evidence.read_text())['emotion_metrics'], 'passed')
            evidence.unlink()
            for key, value in original['emotion_metrics'].items():
                with self.subTest(field=key):
                    altered = copy.deepcopy(original)
                    altered['emotion_metrics'][key] = {'tampered': 1} if isinstance(value, dict) else value + 1
                    (root / 'results_final.json').write_text(json.dumps(altered))
                    rejected = self.run_script(root, 'verify_submission.py')
                    # Clean up bad evidence so each mutation is independently tested in RED.
                    wrote_evidence = evidence.exists()
                    if wrote_evidence:
                        evidence.unlink()
                    self.assertNotEqual(rejected.returncode, 0, f'Tampered {key} accepted')
                    self.assertIn('Emotion metrics mismatch', rejected.stderr)
                    self.assertFalse(wrote_evidence)

    def test_report_zero_denominators_render_na(self):
        for title, covered, tied, covered_text in (
                ('unmatched', 0, 0, '0/0 (N/A)'),
                ('tied', 150, 150, '0/150 (0.0%)'),
                ('happy', 150, 0, '150/150 (100.0%)')):
            with self.subTest(fixture=title), tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                root = Path(tmp)
                report = self.make_fixture(root, title)
                em = report['emotion_metrics']
                self.assertEqual(em['covered'], covered)
                self.assertEqual(em['tied'], tied)
                self.assertEqual(em['unique_top'], 150 if title == 'happy' else 0)
                # A standalone synthetic legacy input, not delivered metrics.
                (root / 'evidence/legacy_binary_summary.json').write_text(json.dumps(dict(
                    correct=1, total=2, accuracy=0.5, baseline=0.5,
                    class_counts=dict(POSITIVE=1, NEGATIVE=1))))
                sources = (ROOT / 'SOURCES.md').read_text()
                (root / 'SOURCES.md').write_text(sources)
                result = self.run_script(root, 'write_report.py')
                self.assertEqual(result.returncode, 0, result.stderr)
                text = (root / 'README.md').read_text()
                self.assertIn('covered reviews agreement is **' + covered_text + '**', text)
                unique_text = '150/150 (100.0%)' if title == 'happy' else '0/0 (N/A)'
                self.assertIn('unique-maximum reviews it is **' + unique_text + '**', text)
                self.assertIn('The public repository belongs to the submitting student’s account', text)
                self.assertTrue(text.endswith(sources))

    def test_optimized_execution_rejected_before_processing(self):
        # Install a fail-closed audit hook before executing each actual script.
        # No input reads, output writes, network attempts, or browser launches allowed.
        bootstrap = '''
import sys
from pathlib import Path
path = Path(sys.argv[1])
code = compile(path.read_text(), str(path), 'exec')
def deny_processing(event, args):
    if event == 'open' or event.startswith('socket.') or event == 'subprocess.Popen':
        print('PROCESSING_ATTEMPT: ' + event, file=sys.stderr)
        raise RuntimeError('Processing forbidden in optimized-mode regression')
sys.addaudithook(deny_processing)
exec(code, {'__name__': '__main__', '__file__': str(path)})
'''
        for name in ('build_dashboard.py', 'run_reviews.py', 'verify_submission.py',
                     'test_dashboard_three.py', 'test_dashboard.py'):
            with self.subTest(script=name), tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                target = Path(tmp) / name
                shutil.copy2(ROOT / name, target)
                before = target.read_bytes()
                result = subprocess.run([sys.executable, '-B', '-O', '-c', bootstrap, str(target)],
                                        cwd=tmp, text=True, capture_output=True, timeout=30)
                self.assertEqual(target.read_bytes(), before)
                self.assertEqual(list(Path(tmp).iterdir()), [target])
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('PROCESSING_ATTEMPT', result.stderr)
                self.assertIn('without -O/-OO', result.stderr)
                self.assertIn('PYTHONOPTIMIZE', result.stderr)


if __name__ == '__main__':
    unittest.main()
