"""Comprobaciones mínimas de la depuración y del modelo. Ejecuta `python test_sistema.py`."""
import unittest

import sistema


class Depuracion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datos = sistema.datos()

    def test_duplicados_eliminados(self):
        resumen = self.datos['resumen']
        self.assertEqual(resumen['duplicados'], 24)
        self.assertEqual(resumen['validos'], resumen['originales'] - resumen['duplicados'])

    def test_sin_valores_invalidos(self):
        for fila in self.datos['filas']:
            self.assertTrue(sistema.es_valido('ap_hi', fila.presion), fila.presion)
            for clave, x in zip(sistema.FEATURES, fila.vector):
                self.assertTrue(sistema.es_valido(clave, x), (clave, x))

    def test_imputacion_usa_la_mediana(self):
        medianas = self.datos['medianas']
        imputadas = [f.presion for f in self.datos['filas'] if f.imputada]
        self.assertTrue(imputadas)
        self.assertEqual(set(imputadas), {medianas['ap_hi']})

    def test_registro_aleatorio_es_real(self):
        for _ in range(50):
            r = sistema.registro_aleatorio()
            self.assertEqual(set(r['datos']), set(sistema.FEATURES))
            self.assertIsInstance(r['presion_real'], float)

    def test_distribucion_cuadra(self):
        d = sistema.distribucion()
        self.assertEqual(sum(i['cantidad'] for i in d['intervalos']), d['total'])
        self.assertEqual(d['total'], len(self.datos['filas']))


class Validacion(unittest.TestCase):
    def test_numero(self):
        self.assertEqual(sistema.numero('12.5'), 12.5)
        for malo in ('', 'x', None, 'nan', 'inf'):
            self.assertIsNone(sistema.numero(malo), malo)

    def test_dentro(self):
        self.assertTrue(sistema.dentro(50, (18, 100)))
        self.assertFalse(sistema.dentro(17, (18, 100)))
        self.assertFalse(sistema.dentro(1.5, (1, 3), entero=True))
        self.assertFalse(sistema.dentro(None, (1, 3)))

    def test_prediccion(self):
        modelo = {'mean': [0] * 9, 'scale': [1] * 9, 'coef': [10] + [0] * 9}
        self.assertEqual(sistema.predict(modelo, [50] * 9), 10)

    def test_etiqueta(self):
        self.assertEqual(sistema.etiqueta(70, 89), '70–<90')
        self.assertEqual(sistema.etiqueta(230, 250), '230–250')


if __name__ == '__main__':
    unittest.main()
