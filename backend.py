from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from langchain.chains.llm import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
)

# load documents
def get_documents(directory_path, glob_pattern):
    loader = DirectoryLoader(directory_path, glob=glob_pattern, show_progress=True, loader_cls=TextLoader, loader_kwargs={"autodetect_encoding":True})
    documents = loader.load()
    return documents


def splits_on_markdown(docs):
    all_sections = []
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    for doc in docs:
        sections = text_splitter.split_text(doc.page_content)
        all_sections.extend(sections)
    return all_sections


def split_recursively(documents):
    chunk_size = 1000
    chunk_overlap = 100
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap, 
                                                    separators=["\n\n", "\n", " ", ""]
                                                    )
    splits = recursive_splitter.split_documents(documents)
    return splits


def get_retreiver(splits, persist_directory,embed_model ,top_k=4):
    if os.path.exists(persist_directory):
        print("Loading index from storage...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
    else:
        print("Creating index...")
        vectordb = Chroma.from_documents(documents=splits, embedding=embed_model, persist_directory=persist_directory)
        vectordb.persist()
    return vectordb.as_retriever(search_kwargs={"k": top_k})


def get_docs(retreiver, query):
    docs = retreiver.get_relevant_documents(query)
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    return docs, combined_text


def rag_chain(chat_llm):
    prompt = ChatPromptTemplate.from_messages(
            [("system","""you are an intelligent virtual assistant who gives accurate information strictly based on the Context Provided. 
                            Carefully understand the user query and provide the information accurately based on the context.  
                            The context will be in markdown format, you should also generate your response in markdown format.  
                            If the text contains image path that ends with ".png" and if they are relevant to the prompt. you must also show that.    
                            If you cannot find the information, you can say "I don't know". 
                            maintain the markdown structure while generating response. 
                            \n\n
                            CONTEXT:\n,
                            \n {context}""" ),
                ("human", "{user_input}"), 
            ], 
        )
    return LLMChain(
            llm = chat_llm,
            prompt=prompt,  
            verbose=False, 
        )


