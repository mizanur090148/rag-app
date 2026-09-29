import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="DocuMind - Smart RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern glassmorphism UI
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main header styling */
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Source citation box */
    .source-box {
        background: rgba(99, 102, 241, 0.08);
        border-left: 3px solid #6366F1;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.85rem;
        color: #334155;
    }
    
    /* Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #DEF7EC;
        color: #03543F;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Check API Key -----------------
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY পাওয়া যায়নি! দয়া করে `.env` ফাইলে আপনার API Key যুক্ত করুন।")
    st.stop()

# ----------------- Imports after validation -----------------
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Define directories
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

# ----------------- Helper Functions -----------------
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

def get_vector_store():
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rag_documents"
    )

def process_uploaded_file(uploaded_file, chunk_size, chunk_overlap):
    """Processes uploaded document, chunks it, and saves to ChromaDB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load document based on extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif file_ext in [".txt", ".md"]:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        # Add metadata source
        for doc in splits:
            doc.metadata["source_name"] = uploaded_file.name

        # Store in ChromaDB
        vector_store = get_vector_store()
        vector_store.add_documents(splits)

        return len(splits)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def clear_vector_store():
    """Clears existing documents from ChromaDB."""
    embeddings = get_embedding_model()
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rag_documents"
    )
    vector_store.delete_collection()
    # Reinitialize empty
    Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rag_documents"
    )

def extract_text_from_content(content):
    """Safely extracts text string from LangChain chunk content (which can be str or list of dicts)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        res = ""
        for item in content:
            if isinstance(item, dict) and "text" in item:
                res += item["text"]
            elif isinstance(item, str):
                res += item
        return res
    return str(content) if content else ""

# ----------------- Sidebar -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/database.png", width=60)
    st.markdown("### ⚙️ কনফিগারেশন ও কন্ট্রোল")
    
    # Model Selection
    model_name = st.selectbox(
        "🤖 AI মডেল নির্বাচন করুন",
        ["gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"],
        index=0
    )
    
    # Chunk Settings
    with st.expander("🛠️ অ্যাডভান্সড সেটিংস"):
        chunk_size = st.slider("Chunk Size", min_value=300, max_value=2000, value=800, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=400, value=150, step=25)
        top_k = st.slider("Top Documents (K)", min_value=1, max_value=8, value=4)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")
    st.markdown("### 📄 ডকুমেন্ট আপলোড")
    uploaded_files = st.file_uploader(
        "PDF বা TXT ফাইল নির্বাচন করুন",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 প্রসেস এবং ইন্ডেক্স করুন", use_container_width=True, type="primary"):
        if uploaded_files:
            total_chunks = 0
            with st.spinner("ডকুমেন্ট প্রসেস এবং ভেক্টর ডাটাবেসে সেভ করা হচ্ছে..."):
                for file in uploaded_files:
                    chunks_count = process_uploaded_file(file, chunk_size, chunk_overlap)
                    total_chunks += chunks_count
            st.success(f"✅ সফলভাবে {len(uploaded_files)} টি ফাইল এবং {total_chunks} টি চাঙ্ক সেভ করা হয়েছে!")
            st.session_state["has_docs"] = True
        else:
            st.warning("আগে অন্তত একটি ফাইল সিলেক্ট করুন।")

    st.markdown("---")
    if st.button("🗑️ ডাটাবেস ক্লিয়ার করুন", use_container_width=True):
        clear_vector_store()
        st.session_state["messages"] = []
        st.session_state["has_docs"] = False
        st.success("ভেক্টর ডাটাবেস সম্পূর্ণ ক্লিয়ার করা হয়েছে!")

# ----------------- Main Chat Area -----------------
st.markdown('<div class="app-title">🧠 DocuMind RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">আপনার ডকুমেন্টের সাথে বুদ্ধিমত্তার সাথে চ্যাট করুন — দ্রুত, সঠিক এবং রেফারেন্স সহ।</div>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "স্বাগতম! বাঁ পাশের প্যানেল থেকে আপনার PDF বা TXT ফাইল আপলোড করে প্রশ্ন করা শুরু করুন।"}
    ]

# Display Existing Chat Messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 ব্যবহৃত রেফারেন্স / সোর্স (Sources)"):
                for idx, src in enumerate(msg["sources"], 1):
                    doc_name = src.get("source_name", "অজানা ডকুমেন্ট")
                    page_num = src.get("page", None)
                    page_info = f" (পৃষ্ঠা: {page_num + 1})" if page_num is not None else ""
                    st.markdown(f"**{idx}. {doc_name}{page_info}:**")
                    st.caption(src.get("content", ""))

# Chat Input
if prompt := st.chat_input("আপনার ডকুমেন্টের বিষয়ে যেকোনো প্রশ্ন লিখুন..."):
    # Add User Message to History
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("ডকুমেন্ট থেকে তথ্য খোঁজা হচ্ছে..."):
            try:
                vector_store = get_vector_store()
                # Similarity search
                results = vector_store.similarity_search_with_relevance_scores(prompt, k=top_k)
                
                # Extract context and sources
                context_texts = []
                sources_data = []
                for doc, score in results:
                    context_texts.append(doc.page_content)
                    meta = doc.metadata.copy()
                    meta["content"] = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                    meta["score"] = round(score, 4) if score is not None else "N/A"
                    sources_data.append(meta)

                context_str = "\n\n---\n\n".join(context_texts) if context_texts else "কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"

                # Prompt Template
                system_prompt = (
                    "You are a helpful, accurate, and professional AI assistant specialized in answering questions based on the provided document context.\n"
                    "Instructions:\n"
                    "1. Answer the question thoroughly based ONLY on the context provided below.\n"
                    "2. If the context does not contain enough information to answer the question, state politely that the document does not contain this information. Do not hallucinate.\n"
                    "3. Respond in the same language as the user's question (e.g. if the user asks in Bengali, answer in Bengali; if in English, answer in English).\n"
                    "4. Structure your answer with clear bullet points or paragraphs when appropriate.\n\n"
                    "Context:\n"
                    "{context}"
                )

                # Conversation history
                history_messages = []
                for m in st.session_state["messages"][-6:-1]:  # Keep last 5 turns
                    if m["role"] == "user":
                        history_messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history_messages.append(AIMessage(content=m["content"]))

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                # Model fallback list if quota is exceeded
                candidate_models = [model_name] + [
                    m for m in ["gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
                    if m != model_name
                ]

                response_placeholder = st.empty()
                full_response = ""
                generation_success = False

                for current_model in candidate_models:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=current_model,
                            google_api_key=api_key,
                            temperature=temperature
                        )
                        chain = prompt_template | llm
                        full_response = ""

                        for chunk in chain.stream({
                            "context": context_str,
                            "chat_history": history_messages,
                            "question": prompt
                        }):
                            text_piece = extract_text_from_content(chunk.content)
                            if text_piece:
                                full_response += text_piece
                                response_placeholder.markdown(full_response + "▌")

                        response_placeholder.markdown(full_response)
                        generation_success = True
                        break
                    except Exception as model_err:
                        err_str = str(model_err)
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                            st.warning(f"⚠️ {current_model} এর কোটা সাময়িকভাবে পূর্ণ, ব্যাকআপ মডেল দিয়ে চেষ্টা করা হচ্ছে...")
                            continue
                        else:
                            raise model_err

                if not generation_success:
                    st.error("সবগুলো মডেলের ফ্রি কোটা সাময়িকভাবে লিমিট অতিক্রম করেছে। দয়া করে ৩০ সেকেন্ড পর পুনরায় চেষ্টা করুন।")

                # Show Sources Expander if sources found
                if sources_data and context_texts:
                    with st.expander("📚 ব্যবহৃত রেফারেন্স / সোর্স (Sources)"):
                        for idx, src in enumerate(sources_data, 1):
                            doc_name = src.get("source_name", "ডকুমেন্ট")
                            page_num = src.get("page", None)
                            page_info = f" (পৃষ্ঠা: {page_num + 1})" if page_num is not None else ""
                            st.markdown(f"**{idx}. {doc_name}{page_info}:**")
                            st.caption(src.get("content", ""))

                # Append assistant response to messages
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources_data if context_texts else []
                })

            except Exception as e:
                st.error(f"দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে: {str(e)}")
