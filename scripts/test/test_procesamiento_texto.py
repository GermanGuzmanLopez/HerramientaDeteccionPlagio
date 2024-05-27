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

    def test_procesar_texto_contracciones(self):
        # Prueba para contracciones
        parrafo = "can't wouldn't"
        esperado = ["can", "not", "would", "not"]
        resultado = procesar_texto(parrafo)
        print(f"Resultado para contracciones: {resultado}")
        self.assertEqual(resultado, esperado)

        # Prueba para versiones sin apóstrofes
        parrafo_sin_contracciones = "cant wouldnt"
        esperado_sin_contracciones = ["cant", "wouldnt"]  # Mantiene las palabras sin cambios si no son reconocidas
        resultado_sin_contracciones = procesar_texto(parrafo_sin_contracciones)
        print(f"Resultado para versiones sin contracciones: {resultado_sin_contracciones}")
        self.assertEqual(resultado_sin_contracciones, esperado_sin_contracciones)

if __name__ == '__main__':
    unittest.main()
