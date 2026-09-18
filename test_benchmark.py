import unittest
from types import SimpleNamespace
from benchmark import normalize, metrics, extract_text

class ScoringTests(unittest.TestCase):
    def test_bengali_normalization(self):
        self.assertEqual(normalize('  আমি\u200d, বাংলা।  '), 'আমি বাংলা')
        self.assertEqual(normalize('২৪টি'), '২৪টি')
    def test_corpus_weighting_and_insertions(self):
        rows=[{'text':'a b c','prediction':'a b'}, {'text':'d','prediction':'d e f'}]
        result=metrics(rows,False)
        self.assertEqual(result['word_edits'],3)
        self.assertEqual(result['wer_percent'],75)
    def test_empty_hypothesis(self):
        self.assertEqual(metrics([{'text':'আমি বাংলা','prediction':''}],True)['wer_percent'],100)
    def test_output_versions(self):
        self.assertEqual(extract_text(([SimpleNamespace(text='আমি')],None),1),['আমি'])
        with self.assertRaises(ValueError): extract_text([],1)
        with self.assertRaises(TypeError): extract_text([None],1)

if __name__=='__main__': unittest.main()
