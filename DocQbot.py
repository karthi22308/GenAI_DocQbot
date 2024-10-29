import streamlit as st
from PyPDF2 import PdfReader
import chromadb
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = chromadb.Client()
collection = client.get_or_create_collection("document_embeddings")


def generate_embeddings(text):
    client = AzureOpenAI()
    res = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    return res.data[0].embedding


def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_document_embeddings(doc_id, text):
    embedding = generate_embeddings(text)
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id]
    )


def retrieve_relevant_chunks(question):
    query_embedding = generate_embeddings(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    relevant_data = "\n".join([doc[0] for doc in results['documents']])

    return relevant_data

def generate_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    client = AzureOpenAI()
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
        max_tokens=256,
        top_p=0.6,
        frequency_penalty=0.7)


    return res.choices[0].message.content


st.title("DocQBot - Automated Document Retrieval.")


uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
if uploaded_file is not None:
    doc_id = uploaded_file.name
    document_text = extract_text_from_pdf(uploaded_file)


    st.write("Processing document and generating embeddings...")
    create_document_embeddings(doc_id, document_text)
    st.success("Document processed and embeddings created successfully!")


question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if uploaded_file is None:
        st.error("Please upload a document first.")
    elif question.strip() == "":
        st.error("Please enter a question.")
    else:

        relevant_data = retrieve_relevant_chunks(question)


        st.write("Generating answer...")
        answer = generate_answer(question, relevant_data)
        st.subheader("Answer:")
        st.write(answer)
