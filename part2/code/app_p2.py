from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import streamlit as st
from htmlTemplates import css, user_template


os.environ["OPENAI_API_KEY"] = "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"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    chunk_size = 500
    splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = chunk_size,
        chunk_overlap = 100,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_input, conversation):        
    response = conversation({'question': user_input})
    #chat_history = response['chat_history']
    st.write(user_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)

if __name__ == '__main__':
    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                conversation = get_conversation_chain(vectorstore)

    st.header("Chat with PDFs :robot_face:")
    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_userinput(user_question, conversation)
