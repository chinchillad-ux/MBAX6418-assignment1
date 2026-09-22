import unittest
import classify

class ThreeClassTests(unittest.TestCase):
    def test_rating_mapping(self):
        self.assertEqual([classify.LABEL_FROM_RATING(x) for x in (1,2,3,4,5)],
                         ['NEGATIVE','NEGATIVE','NEUTRAL','POSITIVE','POSITIVE'])

    def test_balanced_repeatable_full_file(self):
        from collections import Counter
        rows, meta = classify.sample_balanced(classify.DATA, per_class=50, seed=6418)
        again, _ = classify.sample_balanced(classify.DATA, per_class=50, seed=6418)
        self.assertEqual(rows, again)
        self.assertEqual(Counter(r['truth'] for r in rows), {'POSITIVE':50,'NEUTRAL':50,'NEGATIVE':50})
        self.assertEqual(len({r['id'] for r in rows}), 150)
        self.assertEqual(meta['source_rows'], 152410)
        self.assertGreater(max(r['id'] for r in rows), 100000)

    def test_three_class_scoring_and_strict_final_answer(self):
        self.assertEqual(classify.parse_label('Sentiment: NEUTRAL', 'stop'), 'NEUTRAL')
        self.assertEqual(classify.parse_label('Sentiment: POSITIVE', 'length'), 'UNKNOWN')
        self.assertEqual(classify.parse_label('Maybe NEGATIVE then POSITIVE', 'stop'), 'UNKNOWN')
        self.assertEqual(classify.parse_label(None, 'stop'), 'UNKNOWN')
        result = classify.score([{'truth':'POSITIVE','pred':'POSITIVE'}, {'truth':'NEUTRAL','pred':'POSITIVE'}, {'truth':'NEGATIVE','pred':'UNKNOWN'}])
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['unknown'], 1)
        self.assertEqual(result['matrix']['NEUTRAL']['POSITIVE'], 1)
        self.assertAlmostEqual(result['accuracy'], 1/3)
        self.assertEqual(result['per_class']['NEUTRAL']['recall'], 0)

if __name__ == '__main__':
    unittest.main()
