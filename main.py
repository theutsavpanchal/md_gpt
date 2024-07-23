from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from backend import get_documents, splits_on_markdown, split_recursively, get_retreiver, get_docs, rag_chain
from utils import write_doclist, write_response
from dotenv import load_dotenv
import streamlit as st
from io import StringIO
import glob, os
import re

st.set_page_config(
    page_title="MD-GPT",
    page_icon="🔎",
    layout="wide",)

st.markdown("#### 🔎 Markdown-GPT")

if "messages" not in st.session_state:
   st.session_state["messages"] = [{"role":"assistant", "content":"Hello, How can I help you?"}]

if "temp" not in st.session_state:
   st.session_state['temp'] = 0.3


# load env variables
load_dotenv()
GROQ_API =os.getenv('GROQ_API')


hf_embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## global variables
glob_pattern="./*.md"
directory_path = "./knowledge"
persist_directory = "embeddings/hf_chroma"
TOP_K = 2
LLAMA3_70B_INSTRUCT = "llama3-70b-8192"
LLAMA3_8B_INSTRUCT = "llama3-8b-8192"
DEFAULT_MODEL = LLAMA3_8B_INSTRUCT

chat_llm = ChatGroq(
    temperature=0.3,
    model=DEFAULT_MODEL,
    api_key=GROQ_API
)

def get_relevant_text(query):
    documents = get_documents(directory_path, glob_pattern)
    splits = splits_on_markdown(documents) 
    #splits = split_recursively(documents)
    retriever = get_retreiver(splits, persist_directory, hf_embed ,top_k=TOP_K)
    doclist, relevant_text = get_docs(retriever, query)
    write_doclist(doclist)
    return relevant_text

tab1, tab2, tab3 = st.tabs(["Chat with your data", "Relevant Docs", "View Your Docs"])

with tab1:
   with st.container(height=550):
    messages = st.container(height=460)
    for msg in st.session_state.messages:  # show previous messages 
        if msg["role"] == "user":
            messages.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        else:
            messages.chat_message(msg["role"], avatar="🤖").write(msg["content"])
    if query:=st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
        messages.chat_message("user").write(query)
        relevant_text = get_relevant_text(query)
        rag = rag_chain(chat_llm)
        msgs_rag = {
            "context": relevant_text,  
            "user_input": query, 
            }
        generated_response = rag.invoke(msgs_rag)['text']
        st.session_state.messages.append({"role": "assistant", "content": generated_response})
        write_response(generated_response)
        #messages.chat_message("assistant", avatar="🤖").markdown(generated_response, unsafe_allow_html=True)
        lines = generated_response.split('\n')
        for line in lines:
           match = re.match(r'!\[.*\]\((.*)\)', line)
           if match:
               image_path = match.group(1).split(' ')[0]
               if image_path.startswith("./"):
                    image_path="./knowledge/" + image_path[2:]
               messages.image(image_path)
           else:
               messages.markdown(line)
   

with tab2:
   with st.container(height=550):
       st.text(f"Here you can see what documents your query has fetched.\nGenerated inside 'docs' folder \nEvery query will generate {TOP_K} docs,you can change this by modifying TOP_K parameter")
       directory = 'docs'
       txt_files = glob.glob(os.path.join(directory, '*.txt'))
       for i, txt_file in enumerate(txt_files):
           with open(txt_file, 'r', encoding='utf-8') as file:
               content = file.read()
               file.close()
           with st.expander(f"See Doc {i}"):
               st.markdown(content)
    
with tab3:
    uploaded_files= st.file_uploader("Add more markdown docs", type=["md"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            filename = uploaded_file.name
            with open(f"knowledge/{filename}", "w+", encoding="utf-8") as f:
                f.write(string_data)
                f.close()
        

    directory = 'knowledge'
    st.text(f"You can see your original documents here without images")
    md_files = [file for file in os.listdir(directory) if file.endswith('.md')]
    for filename in md_files:
        fp = os.path.join(directory, filename)
        with open(fp, "r", encoding='utf-8') as file:
            content = file.read()
            file.close()
        with st.expander(f"{filename}"):
            st.markdown(content)
        