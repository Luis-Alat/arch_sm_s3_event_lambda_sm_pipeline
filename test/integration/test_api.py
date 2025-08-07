import json
import requests
import sys

def get_api_url(config_path:str):

    with open(config_path, "r") as f:
        outputs = json.load(f)
    
    url_api = outputs["ApiStack"]["ApiGetPredictionUrl"]

    return "https://mock_api_url.com"

if __name__ == "__main__":

    if len(sys.argv) < 3 and "--config" not in sys.argv:
        print("Uso: python test_api.py --config path/al/archivo.json")
        sys.exit(1)

    print("Reading json config")

    config_path = sys.argv[sys.argv.index("--config") + 1]
    api_url = get_api_url(config_path)
    
    print("Api URL extracted succesfully")
    print(f"URL: {api_url}")