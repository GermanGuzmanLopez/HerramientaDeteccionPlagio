import unittest
import os
import pickle
from scripts.deteccion_plagio import analizar_documentos_carpeta, comparar_nuevo_texto, generar_tabla

class TestDeteccionPlagio(unittest.TestCase):

    def setUp(self):
        # Setup de ejemplo para pruebas
        self.carpeta_origen = "./data/textos_origen"
        self.carpeta_test = "./data/test_data"
        self.texto_prueba = "This is a sample text for testing."
        
        # Crear archivos de prueba si no existen
        os.makedirs(self.carpeta_origen, exist_ok=True)
        os.makedirs(self.carpeta_test, exist_ok=True)
        
        with open(os.path.join(self.carpeta_origen, "sample_org.txt"), 'w') as f:
            f.write("This is a sample text for original.")
        
        with open(os.path.join(self.carpeta_test, "sample_test.txt"), 'w') as f:
            f.write(self.texto_prueba)
        
        # Analizar documentos originales
        self.archivos, self.vectores, self.vectorizador = analizar_documentos_carpeta(self.carpeta_origen)
    
    def tearDown(self):
        # Eliminar archivos de prueba después de cada prueba
        os.remove(os.path.join(self.carpeta_origen, "sample_org.txt"))
        os.remove(os.path.join(self.carpeta_test, "sample_test.txt"))

    def test_analizar_documentos_carpeta(self):
        archivos, vectores, vectorizador = analizar_documentos_carpeta(self.carpeta_origen)
        self.assertTrue(len(archivos) > 0)
        self.assertTrue(vectores.shape[0] > 0)
    
    def test_comparar_nuevo_texto(self):
        similitudes = comparar_nuevo_texto(self.texto_prueba, self.archivos, self.vectores, self.vectorizador)
        self.assertTrue(len(similitudes) > 0)
    
    def test_generar_tabla(self):
        tabla = generar_tabla(self.texto_prueba, self.archivos, self.vectores, self.vectorizador, umbral=0.4)
        self.assertTrue(len(tabla) > 0)

if __name__ == '__main__':
    unittest.main()
