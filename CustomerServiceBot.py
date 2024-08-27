from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from pydantic import ValidationError
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import os

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per day", "2 per hour"])


# load_dotenv('secret_keys.env')

app = Flask(__name__)

# api_key = os.getenv('APIKEY')
api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0, openai_api_key=api_key)

file_path = 'data.txt'
loader = TextLoader(file_path, encoding='utf-8')

# Creating the index
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=OpenAIEmbeddings(openai_api_key=api_key)
)
index = index_creator.from_loaders([loader])

@app.route('/')
def home():
    return send_from_directory('templates', 'index.html')

@app.route('/query', methods=['POST'])
def query_index():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Verify method and adjust if necessary
        if hasattr(index, 'query'):
            result = index.query(query, llm=llm)
        else:
            return jsonify({"error": "Index object does not have a 'query' method"}), 500
        
        return jsonify({"result": result})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
