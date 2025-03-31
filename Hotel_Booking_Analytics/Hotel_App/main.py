from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from dotenv import load_dotenv
import os
from rag_bot import rag_chain as create_rag_chain
rag_chain_instance = create_rag_chain()

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers="*") 

api_key = os.getenv("GROQ_API_KEY")

# Print to check (for debugging, remove in production)
print("GROQ API Key:", api_key if api_key else "API Key not found")

# test create_rag_chain
#print(rag_chain_instance.invoke("What is the overall cancellation rate?"))
@app.route('/analytics', methods=['GET'])
def visualize():
    file_path = './data/hotel_analytics.json'
    if not os.path.exists(file_path):
        return jsonify({"error": "Run analytics.py to generate the .json file"}), 404
    with open(file_path, 'r') as f:
        analytics_data = json.load(f)
    return jsonify(analytics_data)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Extract the API key from the Authorization header (or you could extract it from the JSON body)
    api_key = request.headers.get("Authorization")
    if not api_key:
        return jsonify({"error": "No API key provided in the Authorization header"}), 401

    try:
        answer = rag_chain_instance.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, )




