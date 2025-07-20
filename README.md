# CVAnalyzer 🤖📄

**CVAnalyzer** is a Python-based tool that analyzes CVs (PDFs) using modern NLP techniques. It uses Sentence Transformers for chunk embedding, FAISS for semantic search, and either **Mistral** or **Gemma** LLMs (Locally or via Hugging Face) to answer custom questions from the CV.

---

## 🔧 Features

- ✅ Extracts and chunks text from CV PDFs
- ✅ Encodes and caches embeddings to save compute time
- ✅ Uses FAISS for fast vector search
- ✅ Select between **Mistral** or **Gemma** LLMs (toggle with a flag)
- ✅ Answers questions based on the content of the CV
- ✅ Logs system and resource usage with Loguru

---

## 📁 Project Structure

```
CVAnalyzer/
├── CV/                     # Folder for PDF CVs
├── saved_models/          # Embedding and LLM cache
├── main.py                # Main script to run the analyzer
├── cv_analyzer.py         # Class-based logic
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/CVAnalyzer.git
cd CVAnalyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Hugging Face Token

Update your Hugging Face token in `main.py`:

```python
token = "hf_your_access_token"
```

### 4. Run the Tool

```bash
python main.py
```

---

## 🧠 LLM Options

Set the `llm_choice` flag in `main.py`:
- `0` for Mistral (`mistralai/Mistral-7B-Instruct-v0.1`)
- `1` for Gemma (`google/gemma-1.1-7b-it`)

---

## ❓ Sample Questions

```python
questions = [
    "What is the candidate's name?",
    "Does the CV mention Python?",
    "How many years of experience does the candidate have?"
]
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS by Facebook](https://github.com/facebookresearch/faiss)
- [Loguru](https://github.com/Delgan/loguru)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

---
