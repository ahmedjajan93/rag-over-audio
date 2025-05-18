import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transcribe import transcribe_audio  # Make sure this is implemented

# Load API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit config
st.set_page_config(page_title="Ask the Audio", layout="wide")
st.title("🔊 Ask the Audio")

# Session state initialization
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Upload audio
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["mp3", "mp4"])

# Clear state when new file is uploaded
if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded_file:
        # Clear all session state except file tracker
        for key in list(st.session_state.keys()):
            if key != "last_uploaded_file":
                del st.session_state[key]
        st.session_state.last_uploaded_file = uploaded_file.name
        
# Process the uploaded file
if uploaded_file and "transcription" not in st.session_state:
    suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Transcribe
    with st.spinner("🔍 Transcribing audio..."):
        try:
            transcription = transcribe_audio(tmp_path)
            st.success("✅ Transcription complete!")
            st.session_state.transcription = transcription
        except Exception as e:
            st.error(f"Transcription error: {e}")
            st.stop()

# Show transcript
if "transcription" in st.session_state:
    st.subheader("📝 Transcription Preview")
    preview = st.session_state.transcription[:800]
    st.text(preview + ("..." if len(st.session_state.transcription) > 800 else ""))

# RAG pipeline
if "rag_chain" not in st.session_state and "transcription" in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as tf:
        tf.write(st.session_state.transcription)
        tf_path = tf.name

    # Load and split
    loader = TextLoader(tf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Embeddings and retriever
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.
    Be concise and accurate.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Chain
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    st.session_state.rag_chain = rag_chain
    st.session_state.retriever = retriever

# QA interface
if "rag_chain" in st.session_state:
    query = st.text_input("💬 Ask a question about the audio:")
    if st.button("Generate Response", key="ask_button",type="primary") and query:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.rag_chain.invoke({"input": query})
            st.subheader("📌 Answer")
            st.write(response['answer'])

           
else :
    st.info("📤 Upload an audio file to get started.")
