import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import hashlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

# Set page config at the very beginning
st.set_page_config(page_title="Chat PDF")

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PDF_FOLDER_PATH = r"publications"  # Specify the path to your folder with PDFs
FAISS_INDEX_PATH = "faiss_index"
CHECKSUM_FILE_PATH = "checksum.txt"

def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

def get_conversational_chain(template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def calculate_checksum(folder_path):
    hasher = hashlib.md5()
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
    return hasher.hexdigest()

def process_pdfs():
    new_checksum = calculate_checksum(PDF_FOLDER_PATH)
    if not os.path.exists(FAISS_INDEX_PATH) or (os.path.exists(CHECKSUM_FILE_PATH) and open(CHECKSUM_FILE_PATH).read() != new_checksum):
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text_from_folder(PDF_FOLDER_PATH)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            with open(CHECKSUM_FILE_PATH, 'w') as f:
                f.write(new_checksum)
            st.success("Processing complete and FAISS index updated.")
            
def list_paper_titles(docs):
    titles = [doc.metadata.get("title", "Untitled") for doc in docs]
    return "\n".join(titles)

def list_author_papers(author_name, docs):
    papers = []
    for doc in docs:
        if author_name.lower() in doc.page_content.lower():
            title = doc.metadata.get("title", "Untitled")
            summary_template = """
            Provide a summary of the following text:\n\n
            {text}
            """
            summary_chain = get_conversational_chain(summary_template)
            summary_response = summary_chain({"text": doc.page_content}, return_only_outputs=True)
            summary = summary_response.get("text", "No summary found")
            papers.append(f"Title: {title}\nSummary: {summary}\n")
    return "\n".join(papers)

def list_pdf_files_with_keyword(folder_path, keyword):
    pdf_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".pdf") and keyword.lower() in filename.lower()]
    return pdf_files

def list_pdf_files_by_author(folder_path, author_name):
    pdf_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            if author_name.lower() in text.lower():
                pdf_files.append(filename)
    return pdf_files

def user_input(user_question):
    if "list" in user_question.lower() and "papers on" in user_question.lower():
        keyword = user_question.split("list papers on")[-1].strip()
        pdf_files = list_pdf_files_with_keyword(PDF_FOLDER_PATH, keyword)
        if pdf_files:
            pdf_files_text = "\n".join(f"{i+1}. {pdf}" for i, pdf in enumerate(pdf_files))
            st.write("Matching PDF Files:\n", pdf_files_text)
            current_response_text = pdf_files_text
        else:
            st.write("No matching PDF files found.")
            current_response_text = "No matching PDF files found."
    elif "what work" in user_question.lower() or "contributions done by" in user_question.lower():
        author_name = user_question.split("by")[-1].strip()
        pdf_files = list_pdf_files_by_author(PDF_FOLDER_PATH, author_name)
        if pdf_files:
            pdf_files_text = "\n".join(f"{i+1}. {pdf}" for i, pdf in enumerate(pdf_files))
            st.write(f"Papers by {author_name}:\n", pdf_files_text)
            current_response_text = pdf_files_text
        else:
            st.write(f"No papers found for author {author_name}.")
            current_response_text = f"No papers found for author {author_name}."
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if "list" in user_question.lower() and "titles" in user_question.lower():
            titles_text = list_paper_titles(docs)
            st.write("Paper Titles:\n", titles_text)
            current_response_text = titles_text
        elif "contribution" in user_question.lower() or "work by" in user_question.lower():
            author_name = user_question.split("by")[-1].strip()
            author_papers_text = list_author_papers(author_name, docs)
            st.write(f"Papers by {author_name}:\n", author_papers_text)
            current_response_text = author_papers_text
        else:
            context = "\n".join([doc.page_content for doc in docs])

            current_question_template = """
            Based on the provided context, describe the research that has been conducted on the given topic.\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """
            current_chain = get_conversational_chain(current_question_template)

            current_response = current_chain(
                {"context": context, "question": user_question},
                return_only_outputs=True
            )

            current_response_text = current_response.get("text", "No output text found")

            st.markdown("**Reply:**")
            st.write("\n", current_response_text)

            future_question_template = """
            Based on the provided context and the given question, suggest what can be done in the future regarding the topic. Provide detailed and actionable recommendations.\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Future Prospects:
            """
            future_chain = get_conversational_chain(future_question_template)

            future_response = future_chain(
                {"context": context, "question": user_question},
                return_only_outputs=True
            )

            future_response_text = future_response.get("text", "No output text found")

            st.markdown("**Future Prospects:**")
            st.write("\n", future_response_text)

    if 'last_question' not in st.session_state or st.session_state.last_question != user_question:
        st.session_state.history.append({"question": user_question, "reply": current_response_text})
        st.session_state.last_question = user_question

    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 8])
    feedback_col3.write("Provide feedback:")
    feedback_col4, feedback_col5 = st.columns([1, 1])
    if feedback_col4.button("👍"):
        st.write("Thanks for your feedback!")
    if feedback_col5.button("👎"):
        st.write("Sorry to hear that.")

def main():
    st.header("Chat with your Lab, BioMedical Dept, NIT Rourkela🔍📝")

    process_pdfs()

    if 'history' not in st.session_state:
        st.session_state.history = []

    user_question = st.text_area("""Welcome to the BEE Lab chatbot! As an AI assistant, I'm here to help you with any questions or tasks related to the research and activities of the BEE Lab.

My knowledge base covers a wide range of information about the BEE Lab since its inception in 2014. Feel free to ask me specific queries or more open-ended questions - I'll do my best to provide helpful and accurate responses.

How can I assist you today? I'm ready to put my knowledge and capabilities to work for you.""")
    if user_question:
        user_input(user_question)

    st.sidebar.subheader("Chat History")
    if st.session_state.history:
        for i, chat in enumerate(reversed(st.session_state.history[-10:])):
            if st.sidebar.button(f"Topic {i+1}: {chat['question']}", key=f"history_button_{i}"):
                st.session_state.selected_chat = chat

    if 'selected_chat' in st.session_state:
        st.sidebar.write(f"Q: {st.session_state.selected_chat['question']}")
        st.sidebar.write(f"Reply: {st.session_state.selected_chat['reply']}")
st.markdown(
            """
            <style>
        .fixed-text {
            position: fixed;
            right: 10px;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            z-index: 1000;
        }
        .fixed-text-1 {
            bottom: 10px; /* Positioned in the middle bottom */
            left: 50%;
            transform: translateX(-50%);
        }
        .fixed-text-2 {
            bottom: 50px; /* Positioned above the first text */
        }
        .fixed-text-3 {
            bottom: 40px; /* Positioned above the second text */
        }
    </style>
    <div class="fixed-text fixed-text-1">@2024 for Lab,BioMedical Dept., NIT Rourkela</div>
    <div class="fixed-text fixed-text-2">Developed by Nithin & Guna</div>
    <div class="fixed-text fixed-text-3">Mail: nagothigunesh@gmail.com</div>

            """,
            unsafe_allow_html=True
        )
if __name__ == "__main__":
    main()

