from flask import Flask, render_template, request, jsonify
from predict import predict_news

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    error = None

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        if len(text) < 10:
            error = "Please enter at least 10 characters of news text."
        else:
            result = predict_news(text)

    return render_template("index.html", result=result, text=text, error=error)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if len(text) < 10:
        return jsonify({"error": "Please provide at least 10 characters of news text."}), 400

    return jsonify(predict_news(text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
