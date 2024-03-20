from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

os.environ["OPENAI_API_KEY"] = "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"

def get_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        text = ''
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
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

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),vectorstore.as_retriever(), memory=memory)
    return conversation_chain


# def handle_userinput(user_input, conversation):
#     response = conversation({'question': user_input})
#     chat_history = response['chat_history']
#     for i, message in enumerate(chat_history):
#         speaker = 'Bot' if i % 2 == 0 else 'You'
#         print(f'{speaker}: {message.content}')

def handle_userinput(user_input, conversation):
    response = conversation({'question': user_input})
    chat_history = response['chat_history']
    print('Bot:', response['answer'])

if __name__ == '__main__':
    pdf_path = 'Ads cookbook .pdf'
    text = get_pdf_text(pdf_path)
    chunks = get_text_chunks(text)
    embeddings = get_vectorstore(chunks)
    conversation_chain = get_conversation_chain(embeddings)
    print('\n')
    print('############# Enter \'exit\' to Exit the Program #############')
    while True:
        user_input = input('You:')
        if user_input.lower() == 'exit':
            print('Exiting the program...')
            break
        else:
            handle_userinput(user_input, conversation_chain)
            print('\n')
