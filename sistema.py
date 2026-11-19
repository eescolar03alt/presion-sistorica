#!/usr/bin/env python3
"""Demostración educativa de regresión para presión sistólica; sin dependencias externas."""
import csv
import json
import math
import random
import statistics
import traceback
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / 'cardio_train.csv'
MODEL_PATH = BASE / 'modelo.json'
FEATURES = ('age', 'gender', 'height', 'weight', 'cholesterol', 'gluc', 'smoke', 'alco', 'active')
COLS = ('id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio')
# Rangos admitidos por columna; la edad se expresa en años. Un valor fuera de rango se imputa.
RANGOS = {'age': (18, 100), 'gender': (1, 2), 'height': (120, 220), 'weight': (30, 250),
          'cholesterol': (1, 3), 'gluc': (1, 3), 'smoke': (0, 1), 'alco': (0, 1), 'active': (0, 1)}
ENTEROS = {'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'}
RANGO_AP_HI = (70, 250)
RANGO_AP_LO = (30, 200)  # solo sirve para decidir si la sistólica es coherente
TRAMOS = [(70, 89), (90, 109), (110, 129), (130, 149), (150, 169), (170, 189), (190, 209), (210, 229), (230, 250)]

Fila = namedtuple('Fila', 'vector presion imputada')


def etiqueta(lo, hi):
    return f'{lo}–<{hi + 1}' if hi < 250 else f'{lo}–{hi}'


def numero(valor):
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def dentro(x, rango, entero=False):
    return x is not None and rango[0] <= x <= rango[1] and (not entero or x == int(x))


def es_valido(clave, x):
    rango = RANGO_AP_HI if clave == 'ap_hi' else RANGOS[clave]
    return dentro(x, rango, clave in ENTEROS)


def valores(row):
    """Valor numérico de cada columna, o None cuando falta o no es utilizable."""
    edad = numero(row.get('age'))
    v = {'age': edad / 365.25 if edad is not None else None}
    for clave in FEATURES[1:]:
        v[clave] = numero(row.get(clave))
    ap_lo, presion = numero(row.get('ap_lo')), numero(row.get('ap_hi'))
    # Una sistólica menor o igual que la diastólica es imposible; se descarta solo si la
    # diastólica es creíble, porque si está corrupta no hay nada que comparar.
    if dentro(presion, RANGO_AP_HI) and dentro(ap_lo, RANGO_AP_LO) and presion <= ap_lo:
        presion = None
    v['ap_hi'] = presion
    return v


def cargar():
    """Lee el CSV, elimina duplicados e imputa los valores inválidos con la mediana de su columna."""
    crudos, vistos, duplicados = [], set(), 0
    with CSV_PATH.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            # El identificador no cuenta: es único y no aporta información.
            firma = tuple((row.get(k) or '').strip() for k in COLS if k != 'id')
            if firma in vistos:
                duplicados += 1
                continue
            vistos.add(firma)
            crudos.append(valores(row))
    if not crudos:
        raise ValueError('El archivo no contiene registros utilizables.')
    medianas = {}
    for clave in FEATURES + ('ap_hi',):
        buenos = [v[clave] for v in crudos if es_valido(clave, v[clave])]
        if not buenos:
            raise ValueError(f'No hay valores válidos en la columna {clave}.')
        medianas[clave] = statistics.median(buenos)
    filas, imputados = [], 0
    for v in crudos:
        imputada = not es_valido('ap_hi', v['ap_hi'])
        for clave in FEATURES + ('ap_hi',):
            if not es_valido(clave, v[clave]):
                v[clave] = medianas[clave]
                imputados += 1
        filas.append(Fila(tuple(v[k] for k in FEATURES), v['ap_hi'], imputada))
    return {'filas': filas, 'medianas': medianas,
            'resumen': {'originales': len(crudos) + duplicados, 'duplicados': duplicados,
                        'validos': len(filas), 'imputados': imputados,
                        'imputados_ap_hi': sum(f.imputada for f in filas)}}


_DATOS = None


def datos():
    global _DATOS
    if _DATOS is None:
        _DATOS = cargar()
    return _DATOS


def solve(a, b):
    # Eliminación gaussiana con pivoteo parcial.
    n = len(b)
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(a[j][i]))
        if abs(a[pivot][i]) < 1e-12:
            raise ValueError('No se pudo ajustar el modelo con estos datos.')
        a[i], a[pivot] = a[pivot], a[i]
        b[i], b[pivot] = b[pivot], b[i]
        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            for k in range(i, n):
                a[j][k] -= factor * a[i][k]
            b[j] -= factor * b[i]
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        result[i] = (b[i] - sum(a[i][j] * result[j] for j in range(i + 1, n))) / a[i][i]
    return result


def predict(model, v):
    x = [1.0] + [(value - mean) / scale for value, mean, scale in zip(v, model['mean'], model['scale'])]
    return sum(a * b for a, b in zip(model['coef'], x))


def registro_aleatorio():
    filas = datos()['filas']
    # Solo registros con presión medida: comparar contra un valor imputado no informa.
    reales = [f for f in filas if not f.imputada] or filas
    fila = random.choice(reales)
    return {'datos': dict(zip(FEATURES, fila.vector)), 'presion_real': fila.presion}


def distribucion():
    """Cuenta los registros depurados en intervalos de 20 mmHg, sin exponer registros."""
    intervalos = [{'etiqueta': etiqueta(lo, hi), 'cantidad': 0} for lo, hi in TRAMOS]
    total = 0
    for fila in datos()['filas']:
        # Intervalos contiguos también para presiones con decimales.
        indice = min(int((fila.presion - 70) // 20), len(intervalos) - 1)
        intervalos[indice]['cantidad'] += 1
        total += 1
    return {'total': total, 'intervalos': intervalos, **datos()['resumen']}


def train():
    filas = datos()['filas']
    if len(filas) < 100:
        raise ValueError('Se necesitan al menos 100 registros válidos.')
    rows = [(f.vector, f.presion) for f in filas]
    random.Random(42).shuffle(rows)
    cut = int(len(rows) * .8)
    training, test = rows[:cut], rows[cut:]
    mean = [sum(v[i] for v, _ in training) / len(training) for i in range(len(FEATURES))]
    scale = [math.sqrt(sum((v[i] - mean[i]) ** 2 for v, _ in training) / len(training)) or 1 for i in range(len(FEATURES))]
    n = len(FEATURES) + 1
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for v, y in training:
        x = [1.0] + [(v[i] - mean[i]) / scale[i] for i in range(len(v))]
        for i in range(n):
            b[i] += x[i] * y
            for j in range(n):
                a[i][j] += x[i] * x[j]
    for i in range(1, n):
        a[i][i] += 1.0
    model = {'mean': mean, 'scale': scale, 'coef': solve(a, b)}
    preds = [predict(model, v) for v, _ in test]
    errors = [p - y for p, (_, y) in zip(preds, test)]
    baseline = sum(y for _, y in training) / len(training)
    media_prueba = sum(y for _, y in test) / len(test)
    varianza = sum((y - media_prueba) ** 2 for _, y in test)
    por_tramo = []
    for lo, hi in TRAMOS:
        e = [p - y for p, (_, y) in zip(preds, test) if lo <= y <= hi]
        if e:
            por_tramo.append({'desde': lo, 'hasta': hi, 'etiqueta': etiqueta(lo, hi), 'n': len(e),
                              'mae': sum(abs(x) for x in e) / len(e), 'sesgo': sum(e) / len(e)})
    model['metrics'] = {'total_validos': len(rows), 'entrenamiento': len(training), 'prueba': len(test),
                        'mae': sum(abs(e) for e in errors) / len(errors),
                        'rmse': math.sqrt(sum(e * e for e in errors) / len(errors)),
                        'r2': 1 - sum(e * e for e in errors) / varianza if varianza else 0.0,
                        'mae_referencia': sum(abs(baseline - y) for _, y in test) / len(test),
                        'semilla': 42, 'por_tramo': por_tramo,
                        **datos()['resumen']}
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')
    return model


MODEL = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        try:
            self.ruta()
        except Exception:
            self.fallo()

    def do_POST(self):
        try:
            self.ruta()
        except Exception:
            self.fallo()

    def ruta(self):
        if self.path == '/':
            self.enviar(200, (BASE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/metricas':
            metricas = (MODEL or {}).get('metrics')
            if not metricas:
                self.respond(500, {'error': 'El modelo no tiene métricas; vuelve a entrenarlo con --entrenar.'})
            else:
                self.respond(200, metricas)
        elif self.path == '/registro-aleatorio':
            try:
                self.respond(200, registro_aleatorio())
            except (OSError, ValueError):
                self.respond(500, {'error': 'No se pudo cargar un registro válido del archivo.'})
        elif self.path == '/distribucion':
            try:
                self.respond(200, distribucion())
            except (OSError, ValueError):
                self.respond(500, {'error': 'No se pudo cargar la distribución del archivo.'})
        elif self.path == '/predecir' and self.command == 'POST':
            self.predecir()
        else:
            self.send_error(404)

    def predecir(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Solicitud inválida.')
            data = json.loads(self.rfile.read(length))
            v = [float(data[key]) for key in FEATURES]
            if not all(es_valido(k, x) for k, x in zip(FEATURES, v)):
                raise ValueError('Comprueba los valores y rangos del formulario.')
            if not MODEL:
                raise ValueError('El modelo no está disponible.')
            self.respond(200, {'prediccion': round(predict(MODEL, v), 1)})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.respond(400, {'error': 'Comprueba que todos los campos tengan valores válidos.'})

    def enviar(self, status, body, tipo):
        self.send_response(status)
        self.send_header('Content-Type', tipo)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond(self, status, data):
        self.enviar(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')

    def fallo(self):
        # Cualquier error inesperado se registra y se devuelve como 500 en lugar de cerrar la conexión.
        traceback.print_exc()
        try:
            self.respond(500, {'error': 'Error interno del servidor; revisa la terminal donde ejecutas sistema.py.'})
        except OSError:
            pass


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    puerto = 8000
    if '--puerto' in args:
        try:
            puerto = int(args[args.index('--puerto') + 1])
        except (IndexError, ValueError):
            raise SystemExit('Uso: --puerto N (por ejemplo --puerto 8080)')
        if not 1024 <= puerto <= 65535:
            raise SystemExit('El puerto debe estar entre 1024 y 65535.')
    entrenar = '--entrenar' in args or not MODEL_PATH.exists()
    MODEL = train() if entrenar else json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    if entrenar:
        print('Modelo entrenado:', json.dumps(MODEL['metrics'], ensure_ascii=False), flush=True)
    if '--solo-entrenar' not in args:
        print(f'Servidor en marcha: abre http://127.0.0.1:{puerto} en el navegador.\n'
              'Deja esta ventana abierta mientras lo usas; Ctrl+C para detener.', flush=True)
        try:
            ThreadingHTTPServer(('127.0.0.1', puerto), Handler).serve_forever()
        except OSError:
            raise SystemExit(f'El puerto {puerto} está ocupado; usa --puerto con otro valor.')
