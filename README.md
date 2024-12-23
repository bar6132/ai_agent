
# 🚀 AI Project Starter and Builder Agent

An **AI-powered agent** designed to streamline project creation and development, empowering users to start new projects or build them from scratch by analyzing documentation and code files. 

## 🌟 Features

- **File Format Compatibility**: Processes **PDF**, **TXT**, **PY**, **CSV**, **JSON**, and **MD** files seamlessly.
- **Project Initialization**: Reads and understands documentation to suggest or generate project blueprints.
- **Code Generation**: Creates initial implementations based on project requirements.
- **Code Refinement**: Enhances and optimizes existing code for improved quality.

## 🧠 Powered By

This project is built on a robust foundation of **AI frameworks** and **state-of-the-art language models**:

- **Frameworks and Libraries**:
  - [LangChain](https://www.langchain.com/)
  - [TensorFlow](https://www.tensorflow.org/)
- **Large Language Models (LLMs)** via Ollama:
  - **BGE–M3**: File reading and indexing, using [Hugging Face SentenceTransformers](https://www.sbert.net/).
  - **CodeLlama**: For accurate and efficient code generation.
  - **Mistral**: To refine and optimize generated or existing code.

## 📂 Supported File Types

The agent supports multiple file formats, ensuring compatibility with diverse documentation and data:
- **PDF**: Extracts key information from technical documents.
- **TXT**: Reads plain text files with ease.
- **PY**: Analyzes Python scripts for enhancements or reuse.
- **CSV**: Parses data for analysis and integration.
- **JSON**: Handles structured data seamlessly.
- **MD**: Understands and processes Markdown files for documentation or project details.

## 🚀 How It Works

1. **Input Files**: Provide documentation or code files in the supported formats.
2. **AI Analysis**: The agent reads and comprehends the content, extracting essential details.
3. **Project Blueprint**: Generates a clear structure or project outline.
4. **Code Generation**: Produces high-quality, functional code tailored to the project.
5. **Code Refinement**: Leverages advanced models to optimize and polish code.

## 🛠️ Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/bar6132/ai_agent
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-project-starter-agent
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the agent:
   ```bash
   python main.py
   ```

## 📚 Dependencies

- Python 3.8+
- TensorFlow
- LangChain
- Hugging Face SentenceTransformers
- Ollama CLI

## 🤖 Example Usage

```bash
python main.py 
```

- **Input**: Upload a documentation file or an existing codebase.
- **Output**: A fully initialized project or a refined codebase ready for deployment.

## 💡 Acknowledgments

Special thanks to:
- [LangChain](https://www.langchain.com/) for workflow orchestration.
- [TensorFlow](https://www.tensorflow.org/) for its powerful AI ecosystem.
- [Hugging Face](https://huggingface.co/) for their SentenceTransformers.
