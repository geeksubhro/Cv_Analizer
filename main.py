import os
import time
from cv_analyzer import CVAnalyzer

token = "Add_Your_Hugging_Face_Token_Here"  # Hugging Face Token
analyzer = CVAnalyzer(
    llm_choice=0,  # 0 = Mistral, 1 = Gemma
    token=token,
    verbose=False
)

cv_folder = "CV"
questions = [
    "What is the candidate's name?",
    "Does the CV mention Python?",
    "How many years of experience does the candidate have?"
]

# Ensure the CV folder exists
if not os.path.exists(cv_folder):
    os.makedirs(cv_folder)
    print(f"⚠️ Folder '{cv_folder}' was missing. It has been created.")
    print("📂 Please add PDF CV files into the folder and rerun the script.")
    exit()

# Get list of PDF files
pdf_files = [f for f in os.listdir(cv_folder) if f.lower().endswith(".pdf")]

if not pdf_files:
    print(f"⚠️ No PDF files found in the '{cv_folder}' folder.")
    print("📂 Please add at least one CV in PDF format and rerun the script.")
    exit()

# Process each PDF file
for filename in pdf_files:
    pdf_path = os.path.join(cv_folder, filename)
    print(f"\n📄 Processing file: {filename}")

    try:
        # Load or encode the PDF (caching enabled)
        chunks, embeddings = analyzer._load_or_encode(pdf_path)

        # Ask each question and time it
        for q in questions:
            start = time.time()
            answer = analyzer._ask_question_faiss(q, chunks, embeddings)
            duration = time.time() - start
            print(f"Q: {q}\nA: {answer}\n⏱️ Took {duration:.2f} seconds\n")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
