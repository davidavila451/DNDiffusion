from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts import generate_image

app = Flask(__name__)
#CORS(app)


@app.route('/generate_image_py', methods=['POST'])
def generate_image_py():
    if request.is_json:
        data = request.get_json()

        result = data.get('prompt', 0)
        img = generate_image.generate(result)
        return( jsonify({"message": "Data processed successfully", "result": img})), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)