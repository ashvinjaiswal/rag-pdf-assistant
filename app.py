import os
import tempfile

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models # Still need models for recreation

# --- Page Title and Initial Setup ---
st.set_page_config(
    page_title="Streamlit App - RAG PDF Assistant",
    page_icon="👋",
    layout="wide"
)
st.write("# Welcome to your RAG PDF Assistant! 👋")

# Load environment variables
load_dotenv()

# --- Initialize Session State Variables ---
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False
if "messages_expander" not in st.session_state:
    st.session_state.messages_expander = []
if "uploaded_pdf_name" not in st.session_state:
    st.session_state.uploaded_pdf_name = None

client = OpenAI()


# Embedding Model
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Qdrant client for health check and collection management
qdrant_client = QdrantClient(host="localhost", port=6333)

# --- CORRECTED Qdrant Service Status Check ---
st.sidebar.header("App Status")
try:
    # Attempt a simple operation that requires connection, e.g., getting collections
    # This will raise an exception if Qdrant is not running or accessible
    qdrant_client.get_collections()
    st.sidebar.success("All services running!")
except Exception as e:
    st.sidebar.error(f"Could not connect to Qdrant. Please ensure it's running. Error: {e}")
    st.stop() # Stop the app if Qdrant isn't available

if st.session_state.vector_store_initialized:
    st.sidebar.success(f"PDF '{st.session_state.uploaded_pdf_name}' processed!")
    st.sidebar.write("You can now ask questions about the PDF.")
else:
    st.sidebar.info("Upload a PDF and click 'Prepare the PDF' to start.")

# --- Rest of your code remains the same ---
# Step 1: Upload PDF
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)

if pdf_file is not None:
    file_size_mb = pdf_file.size / (1024 * 1024)
    # st.info(f"Uploaded file: **{pdf_file.name}** ({file_size_mb:.2f} MB)")

    if st.session_state.uploaded_pdf_name != pdf_file.name:
        st.session_state.vector_store_initialized = False
        st.session_state.messages_expander = []
        st.session_state.uploaded_pdf_name = pdf_file.name
        st.rerun() # Use st.rerun instead of st.experimental_rerun for newer Streamlit versions

# Step 2: Prepare the PDF button
if st.button("Prepare the PDF", type="primary", use_container_width=True):
    if pdf_file is None:
        st.warning("Please upload a PDF file before preparing it.")
    else:
        with st.spinner("Processing PDF and will start chat soon...", show_time=True):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_path = tmp_file.name

                # Load PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                # Chunk PDF
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
                split_docs = text_splitter.split_documents(documents)

                # Create vector store
                collection_name = "pdf_vector_db"
                qdrant_client.delete_collection(collection_name)
                vector_store = QdrantVectorStore.from_documents(
                    documents=split_docs,
                    url="http://localhost:6333",
                    collection_name=collection_name,
                    embedding=st.session_state.embeddings_model
                )

                st.session_state.vector_store_initialized = True
                st.session_state.messages_expander = [
                    {"role": "assistant", "content": f"Hi! I've processed your PDF: '{st.session_state.uploaded_pdf_name}'. What would you like to know?"}
                ]
                st.success("PDF processed and ready for chat!")

            except Exception as e:
                st.error(f"An error occurred during PDF processing: {e}")
                st.session_state.vector_store_initialized = False
                st.session_state.messages_expander = []
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- Chat Interface ---
if st.session_state.vector_store_initialized:
    st.markdown("---")
    with st.expander("Chat with your PDF", expanded=True):
        for message in st.session_state.messages_expander:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Please ask your question...", key="expander_chat_input"):
            st.session_state.messages_expander.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.spinner("Thinking..."):
                try:
                    # Reconnect to Qdrant for similarity search (no recreation here)
                    vector_db_retriever = QdrantVectorStore.from_existing_collection(
                        url="http://localhost:6333",
                        collection_name="pdf_vector_db",
                        embedding=st.session_state.embeddings_model
                    )

                    search_results = vector_db_retriever.similarity_search(
                        query=prompt, k=5
                    )

                    context = "\n\n".join([
                        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nSource: {result.metadata.get('source', 'N/A')}"
                        for result in search_results
                    ])

                    CHAT_MODEL = "gpt-3.5-turbo"

                    SYSTEM_PROMPT = f"""
                        You are a helpful AI Assistant who answers user queries based solely on the provided PDF document context.
                        Ensure your answer is directly supported by the context.
                        If the answer is not found in the context, explicitly state "I don't have enough information in the provided document to answer that."
                        When referencing information, always cite the page number(s) from the 'Page Number' field in the context where the information was found.
                        Organize your response clearly and concisely.

                        Context:
                        {context}
                    """

                    chat_completion = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=1000
                    )
                    assistant_response = chat_completion.choices[0].message.content

                except Exception as e:
                    assistant_response = f"An error occurred while generating a response: {e}"
                    st.error(assistant_response)

                st.session_state.messages_expander.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.write(assistant_response)

else:
    st.info("Upload a PDF and click 'Prepare the PDF' to start chatting!")