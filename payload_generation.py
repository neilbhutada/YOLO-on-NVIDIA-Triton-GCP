import cv2, base64, json

images = ["<list of image file names>"]
b64 = []
for p in images:
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Encode failed: {p}")
    b64.append(base64.b64encode(enc.tobytes()).decode("ascii"))


with open("payload.json", "w") as f:
    json.dump({
        "inputs": [{
            "name": "images",
            "shape": [len(b64), 1],
            "datatype": "BYTES",
            "data": b64
        }]
    }, f)
