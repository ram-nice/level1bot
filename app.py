import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_conversation_chain(vectorstore):
   llm = ChatOpenAI()
   memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
   conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.as_retriever,
      memory=memory
   )
   return conversation_chain

#chat gpt depenency
def get_vectorstore(text_chunks):
  embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore
   
def get_text_chunks(raw_text):
   text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
      )
   return text_splitter.split_text(raw_text)
   
def get_pdf_text(pdf_doc):
  text =""
  pdf_reader = PdfReader(pdf_doc) 
  for page in pdf_reader.pages:
     text += page.extract_text()
  return text   

def main():
  load_dotenv(os.path.join('dotenv', '.env'))
  st.set_page_config(page_title="Level1 Bot", page_icon=":books:")
  if "conversation" not in st.session_state:
     st.session_state.conversation = None

  st.header(":books: Level1 Bot")
  st.text_input("Feel free to ask anydoubt you have related to the product")

  with st.sidebar:
     st.subheader("Your Documents")
     pdf_doc =  st.file_uploader("Upload your pdfs and click on 'Process'")

     if st.button("Process"):
        with st.spinner("processing"):
         #getPDFTEXT
         raw_text = get_pdf_text(pdf_doc)
        #getTextChunks
        chunks = get_text_chunks(raw_text)
        #createVectorStore
        vectorstore = get_vectorstore(chunks)
        #createConversation
        st.session_state.conversation = get_conversation_chain(vectorstore)
if __name__ == "__main__":
    main()