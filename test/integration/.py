import json
import requests
import sys

def load_api_url(config_path: str) -> str:
    with open(config_path) as f:
        outputs = json.load(f)
    try:
        api_url = outputs["ApiStack"]["ApiGetPrediction"]
        # Asegura que termine en "/"
        if not api_url.endswith("/"):
            api_url += "/"
        return api_url + "predict"
    except KeyError:
        print("❌ No se encontró la URL de la API en el archivo de configuración")
        sys.exit(1)

def run_test(api_endpoint: str):
    payload = {
        "feature1": 1.5,
        "feature2": 3.2
    }

    print(f"👉 Enviando solicitud POST a: {api_endpoint}")
    response = requests.post(api_endpoint, json=payload)

    if response.status_code != 200:
        print(f"❌ Código de estado inesperado: {response.status_code}")
        print("Respuesta:", response.text)
        sys.exit(1)

    try:
        result = response.json()
    except json.JSONDecodeError:
        print("❌ La respuesta no es un JSON válido")
        sys.exit(1)

    if "prediction" not in result:
        print("❌ El campo 'prediction' no está en la respuesta:", result)
        sys.exit(1)

    print("✅ Prueba exitosa. Predicción recibida:", result["prediction"])

if __name__ == "__main__":
    if len(sys.argv) < 3 and "--config" not in sys.argv:
        print("Uso: python test_api.py --config path/al/archivo.json")
        sys.exit(1)

    config_path = sys.argv[sys.argv.index("--config") + 1]
    api_url = load_api_url(config_path)
    run_test(api_url)
