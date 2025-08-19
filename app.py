import os
import traceback
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from a .env file
load_dotenv()

# --- Custom CSS for a modern, clean UI ---
st.markdown("""
<style>
    /* Import Poppins font for a clean, professional look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* Apply font to the whole application and set the primary background */
    .stApp {
        background-color: #f0f2f6; /* A very light grey background */
        font-family: 'Poppins', sans-serif;
    }

    /* Style the main content block for a card-like effect */
    .st-emotion-cache-1c5c0d6 {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
    }

    /* Adjust padding for a cleaner layout */
    .st-emotion-cache-13ln4j9 {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Main header styling */
    h1 {
        color: #4A4A4A; /* Dark gray for a professional tone */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Secondary header for the chat title */
    h2 {
        color: #FF7043; /* Bright orange for emphasis */
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Style for the user's chat message bubble */
    .st-emotion-cache-1629p2d {
        background-color: #4A4A4A !important;
        color: white !important;
        border-radius: 12px 12px 0 12px !important;
        padding: 10px 15px !important;
        margin-right: 15% !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Style for the assistant's chat message bubble */
    .st-emotion-cache-r423a6 {
        background-color: #FF7043 !important;
        color: white !important;
        border-radius: 12px 12px 12px 0 !important;
        padding: 10px 15px !important;
        margin-left: 15% !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Styling for the chat input text area */
    .st-emotion-cache-1d9w38z {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #ddd;
    }

    /* Expander for sources */
    .st-emotion-cache-vof52r {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Use st.secrets for secure API key management, falling back to environment variables.
try:
    groq_api_key = st.secrets["GEMINI_API_KEY"]   # Streamlit Cloud secrets
except KeyError:
    groq_api_key = os.getenv("GEMINI_API_KEY")    # Local .env fallback

if not groq_api_key:
    st.error("⚠️ Missing GEMINI_API_KEY. Please set it in your environment variables or in .streamlit/secrets.toml")
    st.stop()

# Configure the Generative AI model
genai.configure(api_key=gemini_api_key)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate

# --- Session State Initialization ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load FAISS on startup ---
if not st.session_state.retriever:
    st.toast("⏳ Loading vector store...")
    try:
        INDEX_DIR = "faiss_index"
        vectorstore = FAISS.load_local(INDEX_DIR, st.session_state.embeddings, allow_dangerous_deserialization=True)
        st.session_state.retriever = vectorstore.as_retriever()
        st.toast("✅ Vector store loaded successfully!", icon="🎉")
    except Exception as e:
        st.error("⚠️ No vector store found. Please run ingest.py first to generate embeddings.")
        st.error(f"Error details: {e}")
        st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="ML Club Chatbot", layout="wide")

# Updated Title and Header Section
st.markdown("""
<div style="text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
    <h1 style="color: #4A4A4A; font-weight: 700;">ML CLUB CHATBOT</h1>
    <p style="color: #6C757D; font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 2rem;">
        <span style="font-size: 1.5em; margin-right: 5px;">🤖</span>Your intelligent assistant for documents.
    </p>
</div>
<div style="text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 1rem; margin-bottom: 1.5rem;">
    <h2 style="color: #FF7043; font-weight: 600;">💬 Let's get started!</h2>
</div>
""", unsafe_allow_html=True)

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User input ---
user_query = st.chat_input("Ask a question ...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                docs = st.session_state.retriever.get_relevant_documents(user_query)
                context = "\n\n".join(doc.page_content for doc in docs)[:3000]

                prompt_template = ChatPromptTemplate.from_template(
                    """
                    You are an AI Assistant that answers questions about policy documents.
                    Be concise, clear, and say 'I don't know' if unsure.

                    CONTEXT: {context}
                    QUESTION: {question}

                    Answer:
                    """
                )
                final_prompt = prompt_template.format(context=context, question=user_query)

                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(final_prompt)

                answer = response.text.strip()
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.expander("📄 Sources"):
                    for doc in docs:
                        source = doc.metadata.get('source', 'Unknown')
                        st.markdown(f"- **Source:** `{source}`")

            except Exception as e:
                error_message = f"❌ An error occurred: {e}"
                st.error(error_message)
                print(f"Full traceback:\n{traceback.format_exc()}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})
