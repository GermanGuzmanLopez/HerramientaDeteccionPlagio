import unittest
from scripts.procesamiento_texto import procesar_texto

class TestProcesamientoTexto(unittest.TestCase):

    def test_procesar_texto(self):
        parrafo = "Hello, World!"
        esperado = ["hello", "world"]
        resultado = procesar_texto(parrafo)
        self.assertEqual(resultado, esperado)

    def test_procesar_texto_lematizacion(self):
        parrafo = "running runs ran"
        esperado = ["run", "run", "run"]
        resultado = procesar_texto(parrafo)
        self.assertEqual(resultado, esperado)


if __name__ == '__main__':
    unittest.main()
