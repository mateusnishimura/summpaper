import re
import os
import fitz  # PyMuPDF
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from time import time

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
            f.write(f"\n=== Página {page_number + 1} ===\n\n")
            for doc in page_docs:
                content = pipeline_process(doc.page_content)
                f.write(content + "\n\n")
            print(f"Texto da página {page_number + 1} salvo em {output_txt}")
    return output_txt

def load_pdf(pdf_path: str):
    if pdf_path and os.path.exists(pdf_path):
        loader = UnstructuredLoader(file_path=pdf_path)
        pdf = loader.load()
        print(f"PDF carregado com sucesso: {pdf_path}")
        return pdf
    else:
        print("Caminho inválido!")
        return None

# divide text into chunks
def divide_into_chunks(output_path: str):
    with open(output_path, "r", encoding="utf-8") as f:
        txt = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(txt)
    print(f"Texto dividido em {len(chunks)} chunks")
    return chunks

# create vector db
def vector_db(chunks, persist_dir="./chroma_db", collection_name="local-rag"):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    
    if os.path.exists(persist_dir):
        print(f"Carregando banco de dados vetorial existente de {persist_dir}")
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_dir
        )
    else:
        print(f"Criando novo banco de dados vetorial em {persist_dir}")
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        vector_db.persist()
        print(f"Banco de dados vetorial criado e salvo em {persist_dir}")
    
    return vector_db

def get_retriever(vector_db):
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 5}
    )
    return retriever

def generate_structured_summary(vector_db):
    llm = Ollama(model="llama3")
    
    section_prompts = {
        "Introdução": "Analise o contexto fornecido e resuma a introdução do artigo, destacando o objetivo principal da pesquisa, a motivação por trás do estudo e as questões ou hipóteses que orientam o trabalho. Inclua também uma breve menção à abordagem metodológica, se relevante, e ao contexto geral em que a pesquisa se insere:\n\n{context}",
        "Contexto": "Com base no contexto fornecido, descreva detalhadamente o problema ou cenário abordado pelo artigo. Inclua informações sobre o contexto teórico, prático ou social que justifica a pesquisa, os desafios existentes e por que esse problema é relevante para a área de estudo. Destaque também eventuais lacunas no conhecimento que o artigo busca preencher:\n\n{context}",
        "Resultados": "Extraia e resuma os principais resultados do artigo, destacando dados quantitativos ou qualitativos relevantes. Organize as descobertas de forma clara, mencionando tendências, padrões ou insights significativos. Se aplicável, inclua comparações com estudos anteriores ou metas estabelecidas na pesquisa:\n\n{context}",
        "Conclusão": "Resuma as conclusões do artigo de forma concisa, focando nas implicações práticas, teóricas ou sociais dos resultados. Destaque as principais mensagens finais, contribuições para a área de estudo e possíveis direções para pesquisas futuras. Se relevante, mencione limitações do estudo e como elas podem ser superadas:\n\n{context}",
        "Relevância": "Discuta a relevância do artigo de forma abrangente, considerando seu impacto teórico, prático e social. Explique como os resultados ou conclusões contribuem para o avanço da área de estudo, se abordam lacunas existentes e quais são as possíveis aplicações ou implicações dos achados. Além disso, avalie a importância do artigo para diferentes públicos, como acadêmicos, profissionais ou a sociedade em geral:\n\n{context}"
    }
    
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    summary = {}
    for section, template in section_prompts.items():
        start = time()
        
        query = f"Encontre partes do artigo que contenham informações sobre {section.lower()}."
        # get more relevantes sections to the query
        docs = retriever.get_relevant_documents(query)
        context = "\n".join(doc.page_content for doc in docs)

        print(f"Tempo para buscar chunks ({section}): {time() - start:.2f}s")

        # generate summ
        start = time()
        prompt_template = PromptTemplate(input_variables=["context"], template=template)
        prompt = prompt_template.format(context=context)
        summary[section] = llm(prompt)
        print(f"Tempo para LLM ({section}): {time() - start:.2f}s")

    
    formatted_summary = "\n=== Resumo Estruturado do Artigo ===\n"
    for section, content in summary.items():
        formatted_summary += f"\n**{section}**\n{content.strip()}\n"

    return formatted_summary

def interact_with_pdf(retriever, question: str):
    llm = Ollama(model="llama3")
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Com base no seguinte contexto:\n{context}\n\nResponda à pergunta: {question}"
    )
    prompt = prompt_template.format(context=context, question=question)
    answer = llm(prompt)
    return answer

def main(pdf_path, output_path, dir_db):
    pdf_docs = load_pdf(pdf_path)
    if not pdf_docs:
        return
    
    output_path = render_pages(pdf_path, pdf_docs, output_path)

    start = time()
    chunks = divide_into_chunks(output_path)
    print(f"Divisão em chunks:{time() - start}")

    start = time()
    db = vector_db(chunks, dir_db)
    print(f"Criação do banco de dados vetorial:{time() - start}")

    rag_retriever = get_retriever(db)
    print("\nGerando resumo estruturado...")
    summary = generate_structured_summary(rag_retriever)
    print(summary)

    while True:
        question = input("\nFaça uma pergunta sobre o PDF (ou 'sair' para encerrar): ")
        if question.lower() == "sair":
            break
        answer = interact_with_pdf(rag_retriever, question)
        print(f"\nResposta: {answer}")

if __name__ == "__main__":
    pdf_path = "/home/mateus/Downloads/Artigos/Data Labeling for Machine Learning Engineers.pdf"
    output_path = "/home/mateus/Downloads/Artigos/texto_extraido.txt"
    dir_db = "./chroma_db"
    main(pdf_path, output_path, dir_db)