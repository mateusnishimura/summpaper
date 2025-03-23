# 📄 summpaper

## 📝 Project Overview
This project explores an approach that combines **RAG (Retrieval-Augmented Generation)** and **LLMs** to generate **structured summaries of research papers**. The goal is to automatically extract the most relevant passages and organize the summary into the following sections:

- Introduction
- Context
- Results
- Conclusion
- Relevance

## ⚙️ How It Works
1️⃣ **Paper Vectorization** → The document is divided into **chunks** (smaller segments) with overlap to ensure context retention.
2️⃣ **Information Retrieval** → The system retrieves the **most relevant excerpts** based on predefined topics.
3️⃣ **Summary Generation** → **LLaMA 3** processes the retrieved excerpts and generates a **structured and concise summary**.


### ▶️ Running the Script
The main script can be executed as follows:
```bash
python main.py --pdf_file data/sample.pdf --output_file output/summa.txt --database_dir chroma_db
```

### 🛠 Script Parameters
| Parameter          | Description |
|--------------------|------------|
| `-pdf`, `--pdf_file`  | Path to the input PDF file (**required**). |
| `-o`, `--output_file`  | Path to save the extracted text (default: `./summa.txt`). |
| `-db`, `--database_dir` | Directory where **ChromaDB** will be stored (default: `./chroma_db`). |

## 🔧 Future Improvements
- 📌 **Enhance section extraction**, vectorizing chunks by section.
- ⚡ **Optimize LLM response time**.
- 🖼 **Extract images** from papers to enrich summaries.
