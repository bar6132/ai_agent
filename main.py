from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tkinter import Tk, filedialog
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import requests
import warnings
import re
import os

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def pick_file():
    """Open a dialog to pick any file from the local machine."""
    Tk().withdraw()  
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("All Files", "*.*")]  
    )
    if not file_path:
        raise ValueError("No file selected.")
    return file_path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def process_file(file_path):
    """Extract text content from a file (PDF, text, or Python)."""
    _, ext = file_path.rsplit(".", 1)
    ext = ext.lower()

    if ext == "pdf":
        print("Processing PDF...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        full_text = [chunk.page_content.strip() for chunk in chunks]
    
    elif ext == "txt":
        print("Processing Text File...")
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip().split("\n")

    elif ext == "py":
        print("Processing Python File...")
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip().split("\n")

    elif ext in ["csv", "json", "md"]:
        print(f"Processing {ext.upper()} File...")
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip().split("\n")

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not full_text:
        raise ValueError("The selected file is empty or could not be processed.")
    return full_text

def get_bge_m3_embeddings(sentences):
    """Generate embeddings using BGE-M3 model from Sentence-Transformers."""
    model = SentenceTransformer("BAAI/bge-m3")
    embeddings = model.encode(sentences)
    return embeddings

def build_faiss_index(chunks):
    """Build a FAISS index with BGE-M3 embeddings."""
    embeddings = get_bge_m3_embeddings(chunks)
    d = len(embeddings[0])  
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype="float32"))
    return index, chunks


def find_relevant_chunks(query, index, chunks, top_k=3):
    """Find the most relevant chunks for a query."""
    query_embedding = get_bge_m3_embeddings([query])[0]
    distances, indices = index.search(np.array([query_embedding], dtype="float32"), top_k)
    return [chunks[i] for i in indices[0]]


def generate_code_with_codellama(cleaned_doc):
    """Generate Python code using CodeLlama."""
    print("Generating code with CodeLlama...")
    prompt = f"You are Python Developer. Based on the following command, generate code:\n\n{cleaned_doc}, make sure remove unnecessary apostrophe, marks and text, and put the explanation or needed ingo in comment"
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "codellama",
        "prompt": prompt,
        "stream": False  
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        raise RuntimeError(f"Error querying CodeLlama: {response.text}")


def query_mistral(text):
    """Use Mistral model to refine the generated code."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",
        "prompt": f"""Refine and optimize the following Python code,
                      only return the refined code
                      remove any explanation or comment: \n\n{text}""",
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        raise RuntimeError(f"Error querying Mistral: {response.text}")



def save_code_to_file(code):
    """Save Python code to a file with a name chosen by AI."""
    start_index = code.find("```")  
    if start_index != -1:
        code = code[start_index + 9:]  

    file_name_base = code[:10].replace(' ', '_').replace('\n', '').replace('\r', '')  
    file_name_base = re.sub(r'[^a-zA-Z0-9_]', '', file_name_base)  
    
    if not file_name_base:
        file_name_base = "generated_code"
 
    file_name = f"{file_name_base.lower()}_refined.py"
    
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Code exported successfully to {file_name}!")



def main():
    print("=== Multi-Model AI Pipeline with Ollama ===")

    try:
        file_path = pick_file()

        chunks = process_file(file_path)

        index, chunk_data = build_faiss_index(chunks)

        query = input("\nEnter your question or task description: ")
        relevant_chunks = find_relevant_chunks(query, index, chunk_data)
        context = " ".join(relevant_chunks)
        print("\nRelevant Context:\n", context)

        generated_code = generate_code_with_codellama(context)
        print("\nGenerated Code:\n", generated_code)

        refined_code = query_mistral(generated_code)
        print("\nRefined Code:\n", refined_code)

        save_code_to_file(refined_code)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
