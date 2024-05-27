import unittest
import spacy

nlp = spacy.load("en_core_web_sm")

class TestSpacyPipeline(unittest.TestCase):

    def test_spacy_pipeline(self):
        texto = "cannot would not"
        doc = nlp(texto)
        tokens = [token.text for token in doc]
        self.assertEqual(tokens, ["can not", "would", "not"])

if __name__ == '__main__':
    unittest.main()
