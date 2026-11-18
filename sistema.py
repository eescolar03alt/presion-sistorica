#!/usr/bin/env python3
"""Demostración educativa de regresión para presión sistólica; sin dependencias externas."""
import csv
import json
import math
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
FEATURES = ('age', 'gender', 'height', 'weight', 'cholesterol', 'gluc', 'smoke', 'alco', 'active')
MODEL_PATH = BASE / 'modelo.json'


def vector(row):
    return [float(row['age']) / 365.25, float(row['gender']), float(row['height']),
            float(row['weight']), float(row['cholesterol']), float(row['gluc']),
            float(row['smoke']), float(row['alco']), float(row['active'])]


def valid(v, target=None):
    ranges = [(18, 100), (1, 2), (120, 220), (30, 250), (1, 3), (1, 3), (0, 1), (0, 1), (0, 1)]
    if len(v) != len(ranges) or any(not math.isfinite(x) or not lo <= x <= hi for x, (lo, hi) in zip(v, ranges)):
        return False
    if any(v[i] != int(v[i]) for i in (1, 4, 5, 6, 7, 8)):
        return False
    return target is None or (math.isfinite(target) and 70 <= target <= 250)


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
    # Muestreo de reservorio: todos los registros válidos tienen igual probabilidad.
    elegido = None
    cantidad = 0
    with (BASE / 'cardio_train.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            try:
                valores, presion = vector(row), float(row['ap_hi'])
                if not valid(valores, presion):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            cantidad += 1
            if random.randrange(cantidad) == 0:
                elegido = {'datos': dict(zip(FEATURES, valores)), 'presion_real': presion}
    if elegido is None:
        raise ValueError('No hay registros válidos en el archivo.')
    return elegido


def distribucion():
    """Cuenta valores válidos en intervalos de 20 mmHg, sin exponer registros."""
    limites = [(70, 89), (90, 109), (110, 129), (130, 149),
               (150, 169), (170, 189), (190, 209), (210, 229), (230, 250)]
    intervalos = [{'etiqueta': (f'{lo}–<{hi + 1}' if hi < 250 else f'{lo}–{hi}'), 'cantidad': 0} for lo, hi in limites]
    total = 0
    with (BASE / 'cardio_train.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            try:
                valores, presion = vector(row), float(row['ap_hi'])
                if not valid(valores, presion):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            # Intervalos contiguos también para presiones con decimales.
            indice = min(int((presion - 70) // 20), len(intervalos) - 1)
            intervalos[indice]['cantidad'] += 1
            total += 1
    return {'total': total, 'intervalos': intervalos}


def train():
    rows = []
    discarded = 0
    with (BASE / 'cardio_train.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            try:
                v, y = vector(row), float(row['ap_hi'])
                if not valid(v, y):
                    raise ValueError()
                rows.append((v, y))
            except (ValueError, KeyError, TypeError):
                discarded += 1
    if len(rows) < 100:
        raise ValueError('Se necesitan al menos 100 registros válidos.')
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
    errors = [predict(model, v) - y for v, y in test]
    baseline = sum(y for _, y in training) / len(training)
    model['metrics'] = {'total_validos': len(rows), 'descartados': discarded, 'entrenamiento': len(training), 'prueba': len(test),
                        'mae': sum(abs(e) for e in errors) / len(errors),
                        'rmse': math.sqrt(sum(e*e for e in errors) / len(errors)),
                        'mae_referencia': sum(abs(baseline-y) for _, y in test) / len(test), 'semilla': 42}
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')
    return model


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body = (BASE / 'index.html').read_bytes()
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.end_headers(); self.wfile.write(body)
        elif self.path == '/metricas':
            self.respond(200, MODEL['metrics'])
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
        else:
            self.send_error(404)

    def respond(self, status, data):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8'); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if self.path != '/predecir':
            self.send_error(404); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Solicitud inválida.')
            data = json.loads(self.rfile.read(length))
            v = [float(data[key]) for key in FEATURES]
            if not valid(v):
                raise ValueError('Comprueba los valores y rangos del formulario.')
            self.respond(200, {'prediccion': round(predict(MODEL, v), 1)})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.respond(400, {'error': 'Comprueba que todos los campos tengan valores válidos.'})


if __name__ == '__main__':
    import sys
    if '--entrenar' in sys.argv or not MODEL_PATH.exists():
        MODEL = train()
        print('Modelo entrenado:', json.dumps(MODEL['metrics'], ensure_ascii=False), flush=True)
    else:
        MODEL = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    if '--solo-entrenar' not in sys.argv:
        print('Abre http://127.0.0.1:8000 (Ctrl+C para detener)', flush=True)
        ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
