import os
import torch
import pickle
import hashlib
import fitz  # PyMuPDF
import faiss
import time
import numpy as np
import psutil
from tqdm import tqdm
from datetime import datetime
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class CVAnalyzer:
    def __init__(
        self,
        cv_dir="CV",
        model_dir="saved_models",
        token=None,
        chunk_size=300,
        overlap=50,
        llm_choice=0,  # 0 = Mistral, 1 = Gemma
        embedding_model="all-MiniLM-L6-v2",
        verbose=True
    ):
        self.cv_dir = cv_dir
        self.model_dir = model_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token = token
        self.llm_choice = llm_choice
        self.embedding_model_name = embedding_model
        self.verbose = verbose

        os.makedirs(model_dir, exist_ok=True)
        self._setup_logger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._print(f"Running on device: {self.device}")
        self._log_resource_usage("Startup")

        self.embedding_model_path = os.path.join(model_dir, embedding_model)
        self.embedding_model = self._load_embedding_model()

        self.llm_model_name = "google/gemma-1.1-7b-it" if llm_choice == 1 else "mistralai/Mistral-7B-Instruct-v0.1"
        self.llm_model_path = os.path.join(model_dir, self.llm_model_name.split("/")[-1])
        self.llm_pipeline = self._load_llm_pipeline()

    def _print(self, msg):
        if self.verbose:
            print(f"[INFO] {msg}")

    def _setup_logger(self):
        logger.remove()
        log_path = os.path.join(self.model_dir, "cv_analysis.log")
        logger.add(log_path, format="{time} | {level} | {message}", level="INFO")
        logger.info("Logger initialized.")

    def _log_resource_usage(self, step_desc=""):
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.5)
        logger.info(f"[{step_desc}] RAM Usage: {mem.percent}% used, {mem.available / (1024 ** 3):.2f} GB free | CPU Usage: {cpu}%")

    def _log_dir_size(self, dir_path):
        total = 0
        for dirpath, _, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        logger.info(f"Directory '{dir_path}' size: {total / (1024 ** 2):.2f} MB")

    def _load_embedding_model(self):
        self._print("Checking embedding model...")
        logger.info("Checking for embedding model cache...")
        self._log_resource_usage("Before embedding model load")
        model_dir = os.path.join(self.model_dir, self.embedding_model_name)
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            start = time.time()
            self._print(f"Downloading embedding model: {self.embedding_model_name}")
            model = SentenceTransformer(self.embedding_model_name, device=str(self.device))
            model.save(model_dir)
            logger.info(f"Embedding model saved in {time.time() - start:.2f}s.")
        else:
            self._print("Embedding model found in cache.")
            logger.info("Embedding model found in cache.")
        self._log_resource_usage("After embedding model load")
        return SentenceTransformer(model_dir, device=str(self.device))

    def _load_llm_pipeline(self):
        self._print("Checking LLM model...")
        logger.info(f"Checking LLM model in: {self.llm_model_path}")
        self._log_resource_usage("Before LLM model check")
        if not os.path.exists(os.path.join(self.llm_model_path, "config.json")):
            start = time.time()
            self._print(f"Downloading LLM model: {self.llm_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, token=self.token)
            model = AutoModelForCausalLM.from_pretrained(self.llm_model_name, token=self.token)
            tokenizer.save_pretrained(self.llm_model_path)
            model.save_pretrained(self.llm_model_path)
            logger.info(f"LLM model saved in {time.time() - start:.2f}s.")
            self._log_dir_size(self.llm_model_path)
        else:
            self._print("LLM model found in cache.")
            logger.info("LLM model found in cache.")
        self._log_resource_usage("Before LLM pipeline setup")

        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
        model = AutoModelForCausalLM.from_pretrained(self.llm_model_path)
        self._print("LLM model loaded.")
        logger.info("LLM model loaded successfully.")
        self._log_resource_usage("After LLM model load")

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def _extract_text_from_pdf(self, pdf_path):
        self._print(f"Extracting text from: {os.path.basename(pdf_path)}")
        logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
        try:
            doc = fitz.open(pdf_path)
            text = "".join(page.get_text() for page in doc)
            logger.info("PDF text extraction successful.")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def _chunk_text(self, text):
        self._print("Chunking text...")
        logger.info(f"Chunking text (chunk_size={self.chunk_size}, overlap={self.overlap})...")
        self._log_resource_usage("Before text chunking")
        words = text.split()
        chunks = []
        start = 0
        with tqdm(total=len(words), desc="Chunking", leave=False) as pbar:
            while start < len(words):
                end = start + self.chunk_size
                chunks.append(" ".join(words[start:end]))
                advance = self.chunk_size - self.overlap
                start += advance
                pbar.update(min(advance, pbar.total - pbar.n))
        logger.info(f"{len(chunks)} chunks created.")
        self._log_resource_usage("After text chunking")
        return chunks

    def _file_hash(self, file_path):
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_last_modified_time(self, file_path):
        ts = os.path.getmtime(file_path)
        return datetime.fromtimestamp(ts).isoformat()

    def _load_or_encode(self, pdf_path):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        save_path = os.path.join(self.model_dir, base_name)
        os.makedirs(save_path, exist_ok=True)

        meta_file = os.path.join(save_path, "meta.pkl")
        embedding_file = os.path.join(save_path, "embeddings.pkl")
        current_hash = self._file_hash(pdf_path)
        last_modified = self._get_last_modified_time(pdf_path)

        if os.path.exists(meta_file) and os.path.exists(embedding_file):
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
            if meta.get("hash") == current_hash and meta.get("last_modified") == last_modified:
                self._print(f"📄 Using cached embeddings for {base_name}")
                logger.info(f"Using cached embeddings for {base_name}")
                with open(embedding_file, "rb") as f:
                    return pickle.load(f)

        self._print(f"📄 Encoding new embeddings for {base_name}")
        logger.info(f"Encoding new embeddings for {base_name}")
        self._log_resource_usage("Before encoding")
        text = self._extract_text_from_pdf(pdf_path)
        chunks = self._chunk_text(text)
        embeddings = self.embedding_model.encode(chunks)

        with open(embedding_file, "wb") as f:
            pickle.dump((chunks, embeddings), f)
        with open(meta_file, "wb") as f:
            pickle.dump({
                "hash": current_hash,
                "last_modified": last_modified,
                "cached_at": datetime.now().isoformat()
            }, f)

        logger.info(f"Embeddings saved to cache for {base_name}")
        self._log_resource_usage("After encoding and saving embeddings")
        return chunks, embeddings

    def _ask_question_faiss(self, question, chunks, embeddings, top_k=3):
        self._print(f"❓ {question}")
        logger.info(f"Asking question: {question}")
        self._log_resource_usage("Before FAISS search")
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        question_embedding = self.embedding_model.encode([question]).astype("float32")
        _, top_indices = index.search(question_embedding, top_k)
        top_chunks = [chunks[i] for i in top_indices[0]]
        context = " ".join(top_chunks)

        prompt = (
            "You are a helpful assistant. Use only the context below to answer. "
            "If the answer is not there, say 'Information not found in the CV.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        self._log_resource_usage("Before LLM response")
        result = self.llm_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        answer = result[0]["generated_text"].split("Answer:")[-1].strip()
        logger.info(f"Answer: {answer}")
        self._log_resource_usage("After LLM response")
        return answer

    def analyze_cv(self, pdf_path, questions):
        self._print(f"📄 Processing file: {os.path.basename(pdf_path)}")
        logger.info(f"Analyzing CV: {os.path.basename(pdf_path)}")
        chunks, embeddings = self._load_or_encode(pdf_path)
        results = {}
        for question in tqdm(questions, desc="Answering questions", leave=False):
            results[question] = self._ask_question_faiss(question, chunks, embeddings)
        self._print(f"✅ Done analyzing: {os.path.basename(pdf_path)}")
        logger.info(f"Analysis complete for: {os.path.basename(pdf_path)}")
        return results
