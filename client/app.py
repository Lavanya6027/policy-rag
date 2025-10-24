import streamlit as st
import requests
import pandas as pd
import io
from typing import List, Dict, Any

# --- Configuration ---
# NOTE: Replace with your actual FastAPI backend URL
API_BASE_URL = "http://127.0.0.1:8000"

# --- Streamlit Setup ---
st.set_page_config(
    page_title="HR Assistant Lia",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Fixed Header and Styling ---
st.markdown("""
<style>
/* ---------------------------------------------------- */
/* HEADER FIX                                           */
/* ---------------------------------------------------- */
.fixed-header-container {
    position: fixed; 
    top: 0;
    left: 50%;
    transform: translateX(-50%); 
    max-width: 700px; 
    width: 100%;
    z-index: 1000; 
    background-color: var(--background-color); 
    padding-top: 1.5rem; 
    padding-bottom: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
}

/* ---------------------------------------------------- */
/* TOP INPUT FIX (Chat Input at Top) - NEW DESIGN       */
/* ---------------------------------------------------- */

/* Targets the container wrapping the chat input form */
.stChatInputContainer {
    /* Critical: Override Streamlit's default relative/static positioning */
    position: fixed !important; 
    top: 56px !important;  /* Further reduced to tighten gap */
    left: 50% !important;
    transform: translateX(-50%) !important;
    max-width: 700px !important; 
    width: 100% !important;
    z-index: 999;
    /* Add padding/styling to match the aesthetic */
    background-color: var(--background-color); 
    padding: 1rem 0.5rem 0.5rem 0.5rem; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    border-bottom: 1px solid #e0e0e0;
}

/* Welcome message container (above input) */
.welcome-message-container {
    position: fixed;
    top: 48px;  /* Position below header */
    left: 50%;
    transform: translateX(-50%);
    max-width: 700px;
    width: 100%;
    z-index: 998;
    padding: 0 0.5rem;
}

/* Chat messages container with scrollable area */
.chat-messages-container {
    max-height: calc(100vh - 136px);  /* Further reduced to bring messages closer */
    overflow-y: auto;
    padding: 0 0.5rem;
    margin-top: 96px;  /* Further reduced top space for header + input */
    margin-bottom: 2rem;
}

/* Ensure the main content block has proper spacing */
.main .block-container {
    max-width: 700px; 
    padding-top: 0;     /* No top padding needed since input is at top */
    padding-bottom: 0;  /* No bottom padding needed */
}

/* General Styling */
h1 {
    font-size: 24px; 
    margin: 0;
    text-align: center;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Adjust chat message spacing */
.stChatMessage {
    padding: 10px 15px;
    border-radius: 16px; 
    margin-bottom: 8px;
    line-height: 1.4;
    max-width: 95%;
    word-wrap: break-word;
}

/* Auto-scroll functionality - scrolls to top for newest messages */
.auto-scroll {
    scroll-behavior: smooth;
}

/* Custom scrollbar for chat messages */
.chat-messages-container::-webkit-scrollbar {
    width: 6px;
}

.chat-messages-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.chat-messages-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

.chat-messages-container::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}

/* Responsive design adjustments */
@media (max-width: 768px) {
    .fixed-header-container {
        padding-top: 1rem;
        padding-bottom: 0.25rem;
    }
    
    .welcome-message-container {
        top: 44px;
    }
    
    .stChatInputContainer {
        top: 52px !important;
        padding: 0.75rem 0.25rem 0.25rem 0.25rem;
    }
    
    .chat-messages-container {
        margin-top: 90px;
        max-height: calc(100vh - 130px);
    }
    
    h1 {
        font-size: 20px;
    }
}

@media (max-width: 480px) {
    .fixed-header-container {
        padding-top: 0.75rem;
        padding-bottom: 0.25rem;
    }
    
    .welcome-message-container {
        top: 40px;
    }
    
    .stChatInputContainer {
        top: 48px !important;
        padding: 0.5rem 0.25rem 0.25rem 0.25rem;
    }
    
    .chat-messages-container {
        margin-top: 84px;
        max-height: calc(100vh - 124px);
        padding: 0 0.25rem;
    }
    
    h1 {
        font-size: 18px;
    }
}
</style>
""", unsafe_allow_html=True)


# --- State Management (View Control) ---
if 'view' not in st.session_state:
    st.session_state.view = 'chat'

def switch_view(target_view: str):
    """Function to switch between chat and settings views."""
    st.session_state.view = target_view

# --- Utility Functions for API Calls ---

def call_chat_api(query: str) -> Dict[str, Any]:
    """Calls the /chat/ endpoint."""
    url = f"{API_BASE_URL}/chat/"
    try:
        response = requests.post(url, json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API Error: Could not connect or received bad response. ({e})"}

def get_all_resources() -> List[Dict[str, Any]]:
    """Calls the GET /resources endpoint."""
    url = f"{API_BASE_URL}/resources"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch resources: {e}")
        return []

def delete_resource(resource_id: str):
    """Calls the DELETE /resources/{resource_id} endpoint."""
    url = f"{API_BASE_URL}/resources/{resource_id}"
    try:
        response = requests.delete(url)
        response.raise_for_status()
        st.success(f"Resource ID {resource_id} deleted successfully.")
        st.session_state['resources_data'] = get_all_resources() 
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete resource: {e}")

def refresh_corpora():
    """Calls the GET /refresh-corpora endpoint."""
    url = f"{API_BASE_URL}/refresh-corpora"
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        result = response.json()
        if result.get('status') == 'success':
            st.success(result.get('message', "Corpora successfully re-indexed."))
        else:
            st.error(f"Indexing failed: {result.get('message', 'Unknown error.')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to refresh corpora: {e}")

def upload_file(uploaded_file):
    """Calls the POST /resources/upload endpoint."""
    url = f"{API_BASE_URL}/resources/upload"
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(url, files=files)
        response.raise_for_status()
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.info("Remember to click 'Re-index Knowledge Base' for this file to be active.")
        st.session_state['resources_data'] = get_all_resources()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")

# --- UI Components ---

def render_header(title: str, is_settings: bool):
    """Renders the fixed header container (Top part)."""
    
    with st.container():
        st.markdown('<div class="fixed-header-container">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 4, 1])

        with col1:
            if is_settings:
                if st.button("← Back", key="back_btn"):
                    switch_view('chat')
            else:
                st.write(" ") 

        with col2:
            st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)

        with col3:
            if not is_settings:
                if st.button("⚙️", key="settings_btn", help="Open Settings"):
                    switch_view('settings')
            else:
                st.write(" ") 
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_chat_view():
    """Renders the main Chat interface with input at top and messages below."""
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- WELCOME MESSAGE ABOVE INPUT (Fixed Position) ---
    if not st.session_state.messages:
        st.markdown('<div class="welcome-message-container">', unsafe_allow_html=True)
        st.info("👋 Hello! I'm Lia, your HR Assistant. Ask me a question about company policies!")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- CHAT INPUT AT TOP (Fixed Position) ---
    if query := st.chat_input("Ask a question about HR policies...", key="chat_input_main"):
        
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Call the API and get response
        with st.spinner("Lia is thinking..."):
            response_data = call_chat_api(query)
        
        error_message = response_data.get("error")
        if error_message:
            st.error(f"System Error: {error_message}")
            lia_response = f"I'm sorry, an internal error occurred while trying to process your request."
            sources = None
        else:
            lia_response = response_data.get("answer", "I could not find an answer in the policy documents.")
            sources = response_data.get("source_documents", [])

        st.session_state.messages.append({
            "role": "assistant", 
            "content": lia_response, 
            "sources": sources
        })
        st.rerun()
    
    # --- CHAT MESSAGES DISPLAY (Scrollable Area Below Input) ---
    
    # Create a scrollable container for messages
    st.markdown('<div class="chat-messages-container auto-scroll">', unsafe_allow_html=True)
    
    # Display messages in pairs (user query + AI response) with newest pairs first
    messages = st.session_state.messages
    # Group messages into pairs (user + assistant)
    message_pairs = []
    i = 0
    while i < len(messages):
        if i + 1 < len(messages) and messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            # Found a complete pair
            message_pairs.append((messages[i], messages[i + 1]))
            i += 2
        else:
            # Handle orphaned messages (shouldn't happen in normal flow)
            message_pairs.append((messages[i], None))
            i += 1
    
    # Display pairs in reverse order (newest first)
    for user_msg, assistant_msg in reversed(message_pairs):
        # Display user message first
        with st.chat_message(user_msg["role"]):
            st.markdown(user_msg["content"])
        
        # Display assistant response below user message
        if assistant_msg:
            with st.chat_message(assistant_msg["role"]):
                st.markdown(assistant_msg["content"])
                if assistant_msg.get("sources"):
                    with st.expander("📚 Sources Retrieved"):
                        for source in assistant_msg["sources"]:
                            st.caption(f"**{source}**")
    
    # Auto-scroll to top when new messages are added (since newest messages are at top)
    if st.session_state.messages:
        st.markdown("""
        <script>
        // Auto-scroll to top of chat messages (newest messages are at top)
        setTimeout(function() {
            const chatContainer = document.querySelector('.chat-messages-container');
            if (chatContainer) {
                chatContainer.scrollTop = 0;
            }
        }, 100);
        </script>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_settings_view():
    """Renders the Settings panel with resource management tools."""
    
    st.markdown("---")
    
    # 1. Corpora Management
    st.subheader("Knowledge Base Management (RAG)")
    st.caption("Trigger a full re-indexing of the RAG knowledge base. This is required after adding or removing files.")
    if st.button("🔄 Re-index Knowledge Base", key="refresh_btn", type="primary"):
        with st.spinner("Rebuilding vector store and corpora index... (This may take a moment)"):
            refresh_corpora()
            
    st.markdown("---")

    # 2. Upload Resource
    st.subheader("Upload New Policy Document")
    uploaded_file = st.file_uploader("Select a PDF, DOCX, or TXT file to add to the knowledge base:", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file is not None:
        if st.button(f"📥 Upload: {uploaded_file.name}", key="upload_btn"):
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                upload_file(uploaded_file)
                st.rerun() 
                
    st.markdown("---")
    
    # 3. Resource List and Actions (FIXED with inline buttons)
    st.subheader("Current Policy Documents")
    
    if 'resources_data' not in st.session_state:
        st.session_state.resources_data = get_all_resources()
        
    resources_data = st.session_state.resources_data

    if resources_data:
        # Define the header for the custom table layout
        col_name, col_size, col_download, col_delete = st.columns([4, 1.5, 1, 1])
        col_name.markdown("**File Name**")
        col_size.markdown("**Size (KB)**")
        col_download.markdown("**⬇️**")
        col_delete.markdown("**🗑️**")
        st.markdown("---")

        list_container = st.container(border=False) 

        with list_container:
            for i, r in enumerate(resources_data):
                resource_id = r['id']
                file_name = r['file_name']
                size_kb = f"{r['size_bytes'] / 1024:.2f}"
                download_url = f"{API_BASE_URL}/resources/{resource_id}/download"
                
                col_name, col_size, col_download, col_delete = st.columns([4, 1.5, 1, 1])
                
                # 1. File Name
                col_name.markdown(f"`{file_name}`")
                
                # 2. Size
                col_size.markdown(size_kb)
                
                # 3. Download Button
                col_download.link_button("⬇️", download_url, help="Download File", use_container_width=True)
                
                # 4. Delete Button
                delete_key = f"del_{resource_id}"
                
                if col_delete.button("🗑️", key=delete_key, help=f"Delete {file_name}", use_container_width=True):
                    st.session_state[f'confirm_{resource_id}'] = True
                    st.rerun()

                # Confirmation logic that stays on the screen
                if st.session_state.get(f'confirm_{resource_id}', False):
                    st.warning(f"Are you sure you want to delete **{file_name}**? This action cannot be undone.", icon="⚠️")
                    
                    col_spacer, col_confirm, col_cancel = st.columns([4, 1, 1])
                    with col_confirm:
                        if st.button("Confirm Delete", key=f"confirm_exec_{resource_id}", type="primary", use_container_width=True):
                            del st.session_state[f'confirm_{resource_id}']
                            delete_resource(resource_id)
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_exec_{resource_id}", use_container_width=True):
                            del st.session_state[f'confirm_{resource_id}']
                            st.rerun()

                st.markdown("---") # Row separator
                
    else:
        st.warning("No resources found in the knowledge base.")

# --- Application Main Loop ---

main_container = st.container()

with main_container:
    if st.session_state.view == 'chat':
        render_header("HR Assistant Lia 🗣️", is_settings=False)
        render_chat_view()
    elif st.session_state.view == 'settings':
        render_header("Lia Settings ⚙️", is_settings=True)
        render_settings_view()