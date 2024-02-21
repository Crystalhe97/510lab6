from tempfile import NamedTemporaryFile
import os

import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv

from io import BytesIO
from pdfminer.high_level import extract_text

load_dotenv()

st.set_page_config(
    page_title="Resume and Cover Letter Feedback",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload your document for analysis."}
    ]

def analyze_resume(text):
    feedback = ["Resume Feedback:"]
    essential_sections = ["experience", "education", "skills"]
    has_essential_sections = any(section in text.lower() for section in essential_sections)

    if not has_essential_sections:
        feedback.append("Your resume should include essential sections such as Experience, Education, and Skills.")
    if "managed" in text.lower() or "led" in text.lower():
        feedback.append("Good use of action verbs to describe your roles and achievements.")
    else:
        feedback.append("Consider using action verbs to describe your roles and achievements.")
    if len(text.split()) > 500:
        feedback.append("Your resume may be too long. Consider making it more concise.")
    else:
        feedback.append("The length of your resume is appropriate.")

    return " ".join(feedback)

def analyze_cover_letter(text):
    feedback = ["Cover Letter Feedback:"]
    if "dear" not in text.lower():
        feedback.append("Customize your greeting to address a specific person if possible.")
    if "thank you" not in text.lower():
        feedback.append("Consider adding a thank you note towards the end of your letter.")
    if len(text.split()) > 400:
        feedback.append("Your cover letter may be too long. Aim for a concise message that complements your resume.")
    else:
        feedback.append("The length of your cover letter is appropriate.")

    return " ".join(feedback)

def classify_and_analyze_document(text):
    if "experience" in text.lower() and "education" in text.lower():
        return analyze_resume(text)
    elif "dear" in text.lower() or "interview" in text.lower():
        return analyze_cover_letter(text)
    else:
        return "It's challenging to determine if this is a resume or a cover letter. Please ensure the document is relevant to job applications."

    
uploaded_file = st.file_uploader("Upload your resume or cover letter")
if uploaded_file:
    bytes_data = uploaded_file.read()
    text = extract_text(BytesIO(bytes_data))
    feedback = classify_and_analyze_document(text)
    st.write(feedback)
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
        tmp.write(bytes_data)  # write data from the uploaded file into it
        with st.spinner(
            text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."
        ):
            reader = PDFReader()
            docs = reader.load_data(tmp.name)
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                model="gpt-3.5-turbo",
                temperature=0.0,
                system_prompt="You are an expert on the content of the document, provide detailed answers to the questions. Use the document to support your answers.",
            )
            index = VectorStoreIndex.from_documents(docs)
    os.remove(tmp.name)  # remove temp file

    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=False, llm=llm
        )

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history