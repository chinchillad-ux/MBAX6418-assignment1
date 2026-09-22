import unittest
from pathlib import Path

class EmotionTests(unittest.TestCase):
    def test_lexicon_scoring_contract(self):
        import add_emotions as a
        lex={'happy':{'joy':1},'safe':{'trust':1},'bad':{'anger':1,'sadness':1}}
        r=a.word_emotion('HAPPY happy', 'safe', lex)
        self.assertEqual(r['emotion'],'joy')
        self.assertEqual(r['scores']['joy'],2)
        self.assertEqual(r['scores']['trust'],1)
        self.assertEqual(a.word_emotion('safe','happy',lex)['ties'],['joy','trust'])
        self.assertEqual(a.word_emotion('safe','happy',lex)['emotion'],'joy')
        self.assertEqual(a.word_emotion('xyz','',lex)['emotion'],'NONE')

    def test_completed_joint_output_only(self):
        import run_reviews as r
        self.assertEqual(r.parse_output('{"sentiment":"NEUTRAL","emotion":"trust"}','stop'),('NEUTRAL','trust'))
        self.assertEqual(r.parse_output('{"sentiment":"POSITIVE","emotion":"joy"}','length'),('UNKNOWN','UNKNOWN'))
        self.assertEqual(r.parse_output('{"sentiment":"POSITIVE","emotion":"happiness"}','stop'),('UNKNOWN','UNKNOWN'))
        self.assertEqual(r.parse_output(None,'stop'),('UNKNOWN','UNKNOWN'))

if __name__=='__main__': unittest.main()
