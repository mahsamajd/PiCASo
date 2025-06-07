from flask import Flask, request, jsonify
import sys
sys.path.append('/home/icas/Downloads/nubia/nubia_score')
from nubia import Nubia

app = Flask(__name__)
model = Nubia()

@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")
    try:
        score = model.score(text1, text2)
        return jsonify({"score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=9090)
