import streamlit as st
import requests
import os

supported_formats = ("txt", "pdf", "docx", "xlsx")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

with st.sidebar:
    st.title("Upload file")
    st.header("Supported formats")
    st.markdown("""
    * txt
    * pdf
    * docx
    * xlsx
    """)

    uploaded_file = st.file_uploader("Upload file", type=supported_formats)

    if st.button("Upload file"):
        if uploaded_file is not None:
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            with st.spinner("Sending le to backend..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)

                    if response.status_code == 200:
                        st.session_state.session_id = response.json().get("session_id")
                        st.session_state.file_uploaded = True
                        st.session_state.messages = []
                        st.success("File uploaded successfully!")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"Failed to connect to {BACKEND_URL}. Is the backend running?")
        else:
            st.info("Please upload a file in the sidebar to start.")

st.title("RAG agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Write something...", disabled=not st.session_state.file_uploaded):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            answer = "Error..."
            try:
                response = requests.post(f"{BACKEND_URL}/chat", json={"message": prompt,
                                                                      "session_id": st.session_state.session_id})  # return {"response": str}
                answer = response.json().get("response", "No response returned")

            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to {BACKEND_URL}")

            except requests.exceptions.Timeout:
                st.error(f"Timeout on {BACKEND_URL}")

            except requests.exceptions.HTTPError:
                st.error(f"HTTPError on {BACKEND_URL}")

            except Exception as e:
                st.error(f"Error: {e}")
                answer = f"Exception: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
