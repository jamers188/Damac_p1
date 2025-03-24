import streamlit as st
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval import create_retrieval_chain 


st.set_page_config(
    page_title="ChatCUD",
    page_icon="💬",
)

page_config = {
    st.markdown(
    "<h1 style='text-align: center; color: #b22222; font-family: Arial, sans-serif; background-color: #292f4598;'>chatCUD 💬</h1>",
    unsafe_allow_html=True
    ),
    st.markdown("<h4 style='text-align: center; color: white; font-size: 20px; animation: bounce-and-pulse 60s infinite;'>Your CUD AI Assistant</h4>", unsafe_allow_html=True),
}
 
genai.configure(api_key="AIzaSyCy1c3_6TS1vpJy25c43RbYTFcGrhgaaO4")
model = genai.GenerativeModel(model_name="models/gemini-pro")
# Function to extract text from uploaded PDF files
def extract_text(upload):
    text_pdf=''
    for pdf in upload:
        pdf_reader=PdfReader(pdf)
        for pages in pdf_reader.pages:
            text_pdf+=pages.extract_text()
    return text_pdf


# Function to split text into smaller chunks
def get_chunks(text):
    # Initialize a text splitter object with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=200)
    # Split the text into chunks using the text splitter
    chunks = text_splitter.split_text(text)

    return chunks


# # Function to generate embeddings from text chunks and store them
# def get_embeddings_and_store_pdf(chunk_text):
#     # Initialize a Google Generative AI Embeddings model
#     embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     # Create embeddings from the chunk text
#     create_embedding = FAISS.from_texts(chunk_text, embedding=embedding_model)
#     # Save the created embeddings locally
#     create_embedding.save_local("embeddings_index")

# Function to get user input and generate responses based on similarity to stored PDFs
def get_generated_user_input(user_question):
    # Initialize a Google Generative AI Embeddings model
    text_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load stored embeddings from local storage
    stored_embeddings = FAISS.load_local("embeddings_index", text_embedding, allow_dangerous_deserialization=True)
    #Using Retrival Method
    retriever = stored_embeddings.as_retriever(search_type="mmr")
    docs=retriever.get_relevant_documents(user_question)
    return docs

# Function to determine role-based icons
def role_name(role):
    # Assign icons based on the role
    if role == "model":
        return "bot.png"
    elif role == 'user':
        return 'user.png'
    else:
        return None 

def stream(response):
    # Split the response into words and iterate through them, yielding each word with a slight delay
    for word in response.split():
        yield word + " "
        time.sleep(0.04)

# Extracts the user question from pdf prompt in get_generated_user_input() 
def extract_user_question(prompt_response):
    # Iterate through the parts of the prompt response in reverse order
    for part in reversed(prompt_response):
        # Check if the part contains the keyword "Question:"
        if "Question:" in part.text:
            # Split the text after "Question:" and return the extracted user question
            return part.text.split("Question:")[1].strip()
        
def user_input_response(user_question):
    if user_question:
        generated_prompt = get_generated_user_input(user_question)
        prompt = f"You are a AI assistant at the Canadian University Dubai, take information given to you and answer the users question clearly, but also make sure your response is strctured in a readable way and in helpful manner without mentioning anything irrelavent. Make sure to structure it in a helpful way making it easy for the user to read. information: \n{generated_prompt}?\n \nQuestion: \n{user_question}"
        response = st.session_state.chat_history.send_message(prompt)
        return response.text

    
def main():
    with open('dark.css') as f:
        # Apply the CSS style to the page
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 

    start_conversation = model.start_chat(history=[])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = start_conversation
    
    
    for message in st.session_state.chat_history.history:
    # Get the role name of the message and fetch corresponding avatar if available
        avatar = role_name(message.role)
        # Check if avatar exists
        if avatar:
            # Display the message with the role's avatar
            with st.chat_message(message.role, avatar=avatar):
                # Check if the message has 'content' in its parts
                if "Canadian University Dubai" in message.parts[0].text:
                    # Extract the user's question from the message parts (if available)
                    user_question = extract_user_question(message.parts)
                    # Check if a user question is extracted
                    if user_question:
                        # Display the user question using Markdown
                        st.markdown(user_question)
                # Display the message content
                    else:
                        st.markdown(message.parts[0].text)
                else:
                    st.markdown(message.parts[0].text)
    
    user_question = st.chat_input("Ask chatCUD...")

    if user_question is not None and user_question.strip() != "":

        
        with st.chat_message("user", avatar="user.png"):
            st.write(user_question)
        
        responses=user_input_response(user_question)

        if responses:
            with st.chat_message("assistant", avatar="bot.png"):
                st.write_stream(stream(responses))

if __name__ == "__main__":
    main()

