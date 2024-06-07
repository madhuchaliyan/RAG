from flask import Flask, request, jsonify
from flask_cors import CORS
import src.api.rag_api as ragapi

app = Flask(__name__)
CORS(app) 

@app.route('/rag_api', methods=['POST'])
def call_api():
    # Check if the request has JSON data
    if request.is_json:
        # Parse JSON data from the request object
        user_input = request.json['question']
        
        if user_input:
            # Instantiate your RAGAPI class and send the question
            rag = ragapi.RAGAPI()
            response = rag.send_prompt(user_input)
            return jsonify(response), 200  # Return the response as JSON
        else:
            return jsonify({'error': 'Prompt not found in request body'}), 400
    else:
        return jsonify({'error': 'Request must contain JSON data'}), 400


if __name__ == '__main__':
    app.run()
    