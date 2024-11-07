import torch
from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from pyngrok import ngrok
from threading import Thread

# Check if CUDA is available (GPU support)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

DB_FAISS_PATH = 'vectorstore/db_faiss'
MODEL_PATH = 'models/llama_model/model.bin'

app = Flask(__name__)

def set_custom_prompt():
    try:
        prompt = PromptTemplate(
            template="""Use the following pieces of information to answer the user's question.
                        Context: {context}
                        Question: {question}
                        Only return the helpful answer below and nothing else.""",
            input_variables=['context', 'question']
        )
        return prompt
    except Exception as e:
        print(f"Error in set_custom_prompt: {e}")

def load_llm():
    try:
        llm = CTransformers(model=MODEL_PATH, model_type="llama", max_new_tokens=512, temperature=0.5, device=device)
        print("Model loaded successfully:", llm)
        return llm
    except Exception as e:
        print(f"Error in load_llm: {e}")

# Define the QA bot
def qa_bot():
    try:
        print("Running qa_bot")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        response = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': qa_prompt}
        )
        print("QA bot created successfully.")
        return response
    except Exception as e:
        print(f"Error in qa_bot: {e}")

# Initialize the bot once
qa_instance = qa_bot()

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        print("Received:", data)
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Log the qa_instance to confirm it's properly loaded
        print(f"qa_instance: {qa_instance}")
        if not qa_instance:
            return jsonify({'error': 'QA instance not initialized properly'}), 500

        # Run the query through the QA bot
        print("Invoking bot with query:", query)
        response = qa_instance.invoke({'query': query})  # Use 'invoke' instead of '__call__'
        
        # Debugging the response object
        print("Response from bot:", response)

        # Check if the response contains the expected fields
        if 'result' in response:
            answer = response['result']
            sources = [doc.page_content for doc in response.get('source_documents', [])]
            print("Answer:", answer)
            print("Sources:", sources)

            return jsonify({
                'answer': answer,
                'sources': sources
            })
        else:
            print("Error: No 'result' in response")
            return jsonify({'error': 'Failed to get a valid answer'}), 500

    except Exception as e:
        print(f"Error in ask_question: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

def run_flask():
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error in Flask app: {e}")

def run_ngrok():
    try:
        public_url = ngrok.connect(5000)
        print(f" * Flask app running at: {public_url}")
    except Exception as e:
        print(f"Error in Ngrok tunnel: {e}")

if __name__ == '__main__':
    # Start Flask app in a separate thread
    try:
        flask_thread = Thread(target=run_flask)
        flask_thread.start()

        # Start Ngrok tunnel in a separate thread
        ngrok_thread = Thread(target=run_ngrok)
        ngrok_thread.start()
    except Exception as e:
        print(f"Error in starting threads: {e}")
