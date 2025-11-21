import base64
import json
import sys

import requests


def main():
    # Load a sample JPEG as base64
    img_path = "tests/assets/sample.jpg"
    try:
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        msg = (
            "Sample image not found at "
            + img_path
            + ". Please place a JPEG there (e.g., download any small photo)."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    task = {
        "data": {
            "image": "data:image/jpeg;base64," + encoded,
        },
    }

    url = "http://localhost:9090/predict"
    r = requests.post(url, json=task)
    print("PREDICT RESPONSE:", r.status_code)

    try:
        payload = r.json()
    except Exception:
        print("Response is not JSON:", r.text[:500])
        sys.exit(1)

    print(json.dumps(payload, indent=2)[:2000])


if __name__ == "__main__":
    main()
