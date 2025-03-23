import re
import os
import fitz  # PyMuPDF
import json
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from time import time
import argparse
import logging

# Define o nível de log global para WARNING
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("chroma").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
# Desative os logs do Ollama

# join words separated by -
def join_word(text):
    return re.sub(r"-\s+", "", text)

def pipeline_process(content):
    content = join_word(content)
    return content

# render pages and save in .txt
def render_pages(file_path, doc_list: list, output_txt="output.txt") -> str:
    pdf_page = fitz.open(file_path)
    with open(output_txt, "w", encoding="utf-8") as f:
        for page_number in range(pdf_page.page_count):
            page_docs = [
                doc for doc in doc_list if doc.metadata.get("page_number") == page_number
            ]
            for doc in page_docs:
                content = pipeline_process(doc.page_content)
                f.write(content + "\n\n")
            print(f"Extracted page {page_number + 1}")
    return output_txt

def generate_markdown(dicionario):
    emojis = {
    "Introduction": "📝",  
    "Relevance": "💡", 
    "Context": "🌍",      
    "Results": "📊",      
    "Conclusion": "✅"    
    }
    path_summ = os.path.join(os.getcwd(), "summ.md")
    with open(path_summ, "w", encoding="utf-8") as arquivo:
        arquivo.write("# Artigo\n\n")
        
        for secao, resumo in dicionario.items():
            emoji = emojis.get(secao, "📌") 
            arquivo.write(f"## {emoji} {secao}\n\n")
            arquivo.write(f"{resumo}\n\n")

    print(f"Summary saved: '{path_summ}'")

def load_pdf(pdf_path: str):
    if pdf_path and os.path.exists(pdf_path):
        loader = UnstructuredLoader(file_path=pdf_path)
        pdf = loader.load()
        print(f"PDF loaded:{pdf_path}")
        return pdf
    else:
        print("Invalid path")
        return None

# divide text into chunks
def divide_into_chunks(output_path: str):
    with open(output_path, "r", encoding="utf-8") as f:
        txt = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(txt)
    print(f"\nText splitted in {len(chunks)} chunks\n")
    return chunks

# create vector db
def vector_db(chunks, persist_dir, collection_name="local-rag"):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    
    if os.path.exists(persist_dir):
        print(f"Loading db in {persist_dir}")
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_dir,
        )
    else:
        print(f"Creating new db in {persist_dir}")
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        print(f"Created db in {persist_dir}")
    
    return vector_db

def get_retriever(vector_db):
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 5}
    )
    return retriever

def save_json(summary):
    path_summ = os.path.join(os.getcwd(), "summ.json")
    with open(path_summ, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"Saved dict in: {path_summ}")

def generate_structured_summary(retriever):
    llm = OllamaLLM(model="llama3")
    
    section_prompts = {
        "Introduction": "Analyze the provided context and summarize the introduction of the article, highlighting the main objective of the research, the motivation behind the study, and the questions or hypotheses guiding the work. Also include a brief mention of the methodological approach, if relevant, and the general context in which the research is situated:\n\n{context}\n In English",
        "Context": "Based on the provided context, describe in detail the problem or scenario addressed by the article. Include information about the theoretical, practical, or social context that justifies the research, the existing challenges, and why this problem is relevant. Also highlight any gaps in knowledge that the article seeks to fill:\n\n{context}\n In English",
        "Results": "Extract and summarize the main results of the article, highlighting relevant quantitative or qualitative data. Organize the findings clearly, mentioning trends, patterns, or significant insights. If applicable, include comparisons with previous studies or goals established in the research:\n\n{context}\n In English",
        "Conclusion": "Summarize the conclusions of the article concisely, focusing on the practical, theoretical, or social implications of the results. Highlight the key final messages, contributions to the field of study, and possible directions for future research. If relevant, mention limitations of the study:\n\n{context}\n In English",
        "Relevance": "Discuss the relevance of the article, considering its theoretical and practical impact. Explain how the results or conclusions contribute to the advancement of the field, whether they address existing gaps, and what the potential applications or implications of the findings are:\n\n{context}\n In English"
    }
    
    summary = {}
    
    for section, template in section_prompts.items():
        start = time()

        query = f"Find parts of the article that contain information about {section.lower()}."
        docs = retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in docs)

        #print(f"Tempo para buscar chunks ({section}): {time() - start:.2f}s")

        start = time()
        prompt_template = PromptTemplate(input_variables=["context"], template=template)
        prompt = prompt_template.format(context=context)
        summary[section] = llm.invoke(prompt)
        print(f"\n✅ Summary for '{section}' completed!")
        #print(f"\nTempo para LLM ({section}): {time() - start:.2f}s")

    save_json(summary)  
    generate_markdown(summary)

    return summary

def interact_with_pdf(retriever, question: str):
    llm = OllamaLLM(model="llama3")
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Based on the following context:\n{context}\n\nAnswer the question: {question}"
    )
    prompt = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt)
    return answer

def main(pdf_path, output_path, dir_db):
    pdf_docs = load_pdf(pdf_path)
    if not pdf_docs:
        return
    
    output_path = render_pages(pdf_path, pdf_docs, output_path)

    start = time()
    chunks = divide_into_chunks(output_path)
    #print(f"Divisão em chunks:{time() - start}")

    start = time()
    db = vector_db(chunks, dir_db)
    #print(f"Criação do banco de dados vetorial:{time() - start}")

    rag_retriever = get_retriever(db)
    print("\nCreating summary...")
    summary = generate_structured_summary(rag_retriever)
    #print(summary)

    while True:
        question = input("\nAsk a question about the PDF (or type 'exit' to quit):")
        if question.lower() == "quit":
            break
        answer = interact_with_pdf(rag_retriever, question)
        print(f"\nResposta: {answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a PDF and save it to ChromaDB.")

    parser.add_argument("-pdf", "--pdf_file", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("-o", "--output_file", type=str, default=os.path.join(os.getcwd(), "summa.txt"),
                         help="Path to save the extracted text.")
    parser.add_argument("-db", "--database_dir", type=str, default=os.path.join(os.getcwd(), "chroma_db"), 
                        help="Directory where ChromaDB will be stored (default: ./chroma_db).")

    args = parser.parse_args()
    
    main(args.pdf_file, args.output_file, args.database_dir)