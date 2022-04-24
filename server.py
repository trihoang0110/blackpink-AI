from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import util
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# CORS(app, resources={r"/*":{"origins":"*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def classify_image():
    if request.method == 'POST':
        try:
            image_data = request.form['image_data']
            response = util.classify_image(image_data)
            # print(response)
            json_response = json.dumps(response, indent = 4)
            return json_response
        except Exception as e:
            # return jsonify({'idol': None,
            #         'class': None})
            print(e)


if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(debug=True)
