import streamlit as st
from streamlit.components.v1 import iframe
import time
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# Styling for the side bar
st.set_page_config(page_title="LLaMA Chat", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 450px;
        max-width: 450px;
        width: 450px;
        overflow-x: hidden;
        overflow-y: hidden;
    }
    [data-testid="stSidebar"] > div {
        resize: none !important;
    }
    ::-webkit-scrollbar {
        width: 0px;
        background: transparent;
    }

    /* Allow collapsing properly */
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px !important;
        max-width: 0px !important;
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🦙 Talk with YouTube videos!")

# Sidebar - For Upload or Fetching transcript
with st.sidebar:
    st.header("📄 Video Loader")
    input_mode = st.radio("Source:", ["YouTube Video ID", "Upload .txt File"])

    def extract_video_id(url):
        try:
            parsed_url = urlparse(url)
            if "youtube.com" in parsed_url.netloc:
                return parse_qs(parsed_url.query).get("v", [None])[0]
            elif "youtu.be" in parsed_url.netloc:
                return parsed_url.path.strip("/")
        except:
            return None

    def extract_video_title(url):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.title.string.replace(" - YouTube", "").strip()
        else:
            return "Unknown Title"
    
    # Fetch transcript using the YouTubeTranscript API
    if input_mode == "YouTube Video ID":
        video_url = st.text_input("Paste YouTube URL:", value=st.session_state.get("video_url", ""))
        if video_url:
            st.session_state.video_url = video_url
            video_id = extract_video_id(video_url)
            try:
                transcript = YouTubeTranscriptApi().fetch(video_id)
                transcript_text = "\n".join([seg.text for seg in transcript])
                st.session_state.transcript_text = transcript_text
                st.session_state.vectorise = True
                st.session_state.video_title = extract_video_title(video_url)
                st.success("✅ Fetched transcript.")
                st.markdown(
                    f"<p style='font-weight: bold; font-size: 20px;'>Title: {st.session_state.video_title}</p>",
                    unsafe_allow_html=True
                )

                # The code for the embedded video on the left side
                embed_url = f"https://www.youtube.com/embed/{video_id}"
                st.session_state.embed_url = embed_url
                iframe(embed_url, width=400, height=250)
            except Exception as e:
                st.error(f"Failed: {e}")
    
    # The below statement applies when the user wants to upload a trasnscript manually (it can be from any platform)
    else:
        uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
        if uploaded_file:
            transcript_text = uploaded_file.read().decode("utf-8")
            st.session_state.transcript_text = transcript_text
            st.session_state.vectorise = True
            st.success("✅ Uploaded.")
            st.text_area("Preview", transcript_text, height=260)

    # Placeholder for button
    reset_placeholder = st.empty()

    # Custom CSS to pin it to the bottom
    # Inject CSS to make sidebar full height and push button to bottom
    st.markdown("""
        <style>
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .reset-btn-wrapper {
            margin-top: auto;
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Wrap Reset button in a bottom-aligned container
    st.markdown('<div class="reset-btn-wrapper">', unsafe_allow_html=True)
    if st.button("Reset Video Session"):
        st.session_state.pop("transcript_text", None)
        st.session_state.pop("video_url", None)
        st.session_state.pop("video_title", None)
        st.session_state.pop("embed_url", None)
        st.session_state.pop("retriever", None)
        st.session_state.messages.append({"role": "assistant", "content": "---"})
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "retriever" not in st.session_state and st.session_state.get("transcript_text"):
    # Splitting the document into chunks of size 1000
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([st.session_state.transcript_text])

    # Using the embedding model to convert the chunks into vectors
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Storing the vectors into vector store for efficient retrieval
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retrieving the 4 most relevent chunks (documents) form the vector store using semantic similarity
    st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# This PromptTemplate defines the structure for interacting with the LLM.
# It ensures that the assistant answers questions in a friendly, helpful way,
# uses the YouTube video title, transcript/context, and the user’s question as inputs,
# and decides when to rely on the provided context vs. answer more casually.
prompt_template = PromptTemplate(
    template="""You are a friendly and helpful assistant. The user is asking questions about the YouTube video titled: "{title}".

    Your job is to read the context or transcript or video, and answer questions relevant to it. 
    When possible, use the context below to answer. If the context is irrelevant, answer generally.
    If you see words like video, transcript, what, how, when, where, explain, that usually means context is relevant.
    Otherwise, you can ignore the context and reply casually to what is asked, however ask the user to ask questions regarding the video.

    Title: {title}

    Context:
    {context}

    Question:
    {question}""",

    input_variables=["context", "question", "title"]
)

# Used to join the documents after retrieval
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(q):
    return (RunnableParallel({
            "context": st.session_state.retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "title": lambda _: st.session_state.get("video_title", "Unknown Title")
        })
        | prompt_template
        | ChatOllama(model="llama3", temperature=0.7)
        | StrOutputParser())

# Show chat messages
if not st.session_state.messages:
    st.markdown("""
    <div style='background-color: #f5f5f5; padding: 20px; border-radius: 10px; font-size: 1.1rem;'>
        <strong>👋 Hi! I’m your YouTube Transcript Chatbot</strong><br><br>
        <ol style="padding-left: 20px;">
            <li>🎮 To ask questions about a YouTube video, paste its full URL in the left panel</li>
            <li>📄 If you already have a transcript, upload it using the file upload option on the left</li>
        </ol>
        <p><strong>⚠️ Note:</strong> The YouTube video must have subtitles for this to work.</p>
        <p>Start by asking me a question below 👇</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if st.session_state.get("retriever"):
            chain = build_chain(prompt)
            with st.spinner("Thinking..."):
                try:
                    assistant_response = chain.invoke(prompt)
                except Exception as e:
                    assistant_response = f"Error: {e}"
        else:
            assistant_response = "No transcript loaded yet."

        # To show the typing animation
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
