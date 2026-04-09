import streamlit as st
from ai_engine import ask_network_ai

st.set_page_config(page_title="NetDevOps AI Assistant", page_icon="⚡", layout="wide")
st.title("⚡ NetDevOps AI Assistant ")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("🛠️ Diagnostic Tools")
    tool = st.selectbox(
        "Select Operating Mode", 
        ["Chat Assistant", "Log Analyzer", "Config Generator", "Packet Explanation"]
    )
    
    st.markdown("---")
    st.header("📂 Context Upload")
    st.write("Upload a config or log file.")
    uploaded_file = st.file_uploader("Upload File", type=["txt", "log", "conf", "cfg"])
    
    file_context = ""
    if uploaded_file is not None:
        file_context = uploaded_file.read().decode("utf-8", errors="ignore")
        st.success(f"Loaded: {uploaded_file.name}")
        
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input(f"Ask the AI ({tool} mode)..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call the engine and stream the chunks
        for chunk in ask_network_ai(prompt, tool, file_context):
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})