import unittest
import spacy
import contractions

nlp = spacy.load("en_core_web_sm")

class TestSpacyPipeline(unittest.TestCase):

    def test_spacy_pipeline(self):
        texto = "can't wouldn't"
        texto_expandido = contractions.fix(texto)
        doc = nlp(texto_expandido)
        tokens = [token.text for token in doc]
        self.assertEqual(tokens, ["can", "not", "would", "not"])

if __name__ == '__main__':
    unittest.main()
