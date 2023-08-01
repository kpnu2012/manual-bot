import psutil
from flask import Flask, render_template, request, jsonify
from openai.error import RateLimitError
from llama_index import GPTVectorStoreIndex, download_loader, StorageContext, load_index_from_storage, LLMPredictor, ServiceContext

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from langchain import OpenAI


app = Flask(__name__)
process = psutil.Process()
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/gpt3', methods=['GET', 'POST'])
def gpt4():
    user_input = request.args.get('user_input') if request.method == 'GET' else request.form['user_input']
    messages = [{"role": "user", "content": user_input}]

    try:
        # rebuild storage context

        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.8, model_name="text-davinci-003"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
        # load index
        index = load_index_from_storage(storage_context,service_context=service_context)
        query_engine = index.as_query_engine()
        response = query_engine.query("Search all documents for an answer to my question. After you provide an answer to the question, tell me what page of the manual I can find more info on. Answer the question first, then provide the page. Question: " + user_input)
        content = str(response)
        content = content + " Access the manual here: www.wikihow.com/manuals/ABCXYZ"
        print(response.source_nodes)
        print(response.get_formatted_sources())
        if "in the context information" in content:
            content = "I couldn't find an answer to this question. Please rephrase it. Make sure you include the make & model of the product you are using."
    except RateLimitError:
        content = "The server is experiencing a high volume of requests. Please try again later."

    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True)