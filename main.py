import streamlit as st
import requests
import base64
import json
from config import get_settings
from pathlib import Path

settings = get_settings()
META_FILE = Path(settings.META_FILE)

def load_sources():
    try:
        with open(META_FILE) as f:
            return ["All"] + json.load(f)
    except:
        return ["All"]

API_URL = "http://localhost:8000/generate"

st.set_page_config(page_title="Multimodal RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_files" not in st.session_state:
    st.session_state.source_files = load_sources()

# ---------- SIDEBAR ----------
st.sidebar.header("Filters")

# 🔄 Refresh button
if st.sidebar.button("🔄 Refresh files"):
    st.session_state.source_files = load_sources()
    st.sidebar.success("File list updated")

# 📂 Dropdown
selected_file = st.sidebar.selectbox(
    "Select Document",
    st.session_state.source_files
)

# ---------- UI HEADER ----------
st.title("📊 Multimodal RAG Chat")
st.caption("Ask questions related to a PDF document")

# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show images if exist
        if "images" in msg:
            for img in msg["images"]:
                st.image(img, use_container_width=True)

# ---------- USER INPUT ----------
query = st.chat_input("Ask something...")

if query:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # ---------- CALL API ----------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            payload = {"query": query}

            if selected_file != "All":
                payload["filter_modality"] = selected_file

            response = requests.post(API_URL, json=payload)
                

            data = response.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

            st.markdown(answer)

            images = []

            # ---------- TOP-1 SOURCE LOGIC ----------
            top_source = next(
                (s for s in sources if s.get("image_base64")),
                sources[0] if sources else None
            )
            if top_source:
                with st.expander("Top Source"):
                    # Show text
                    if top_source.get("text"):
                        st.write(top_source["text"][:300] + "...")

                    # Show text
                    if top_source.get("text"):
                        st.write(top_source["text"][:300] + "...")

                    # Show image if available
                    if top_source.get("image_base64"):
                        try:
                            img_bytes = base64.b64decode(top_source["image_base64"])
                            st.image(img_bytes, caption="📊 Extracted Visual", use_container_width=True)
                            images.append(img_bytes)
                        except Exception:
                            st.warning("Image rendering failed")

                    # Caption (optional)
                    if top_source.get("caption"):
                        st.caption(top_source["caption"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "images": images
            })


