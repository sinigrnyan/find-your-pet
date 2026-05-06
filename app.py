from flask import Flask, render_template, request, jsonify
from map_logic import generate_map

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    zhiv = data["zhiv"]
    khar = data["khar"]
    lat = float(data["lat"])
    lon = float(data["lon"])
    rad = float(data["rad"])
    vol = int(data["vol"])
    m = generate_map(zhiv, khar, lat, lon, rad, vol)
    return jsonify(m)

if __name__ == "__main__":
    app.run(debug=True)