import unittest
from scripts.vectorizacion import vectorizar_documentos, vectorizar_nuevo_texto

class TestVectorizacion(unittest.TestCase):

    def test_vectorizar_documentos(self):
        textos = ["sample text one", "sample text two"]
        vectores, vectorizador = vectorizar_documentos(textos)
        self.assertEqual(vectores.shape[0], 2)
    
    def test_vectorizar_nuevo_texto(self):
        textos = ["sample text one", "sample text two"]
        _, vectorizador = vectorizar_documentos(textos)
        nuevo_texto = "sample text three"
        nuevo_texto_vector = vectorizar_nuevo_texto(nuevo_texto, vectorizador)
        self.assertEqual(nuevo_texto_vector.shape[0], 1)

if __name__ == '__main__':
    unittest.main()
