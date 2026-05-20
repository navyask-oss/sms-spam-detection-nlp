from flask import Flask, jsonify, request

from predict import predict_message


app = Flask(__name__)


@app.get("/")
def health_check():
    return jsonify({"status": "ok", "message": "SMS spam detector is running."})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "").strip()

    if not message:
        return jsonify({"error": "`message` is required."}), 400

    result = predict_message(message)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
