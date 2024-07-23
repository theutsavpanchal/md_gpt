# Markdown-GPT
RAG chain for Markdown files.  
Still in development.

### Requirements 
Conda: 24.5.0

Info:
- **Embeddings by Hugging Face Embeddings: model- all-MiniLM-L6-v2**
- **Add your markdown files inside "/knowledge" directory or directly upload from the UI.**  
- **Relevant Docs are generated inside "/docs" directory"**  

### How to run 
1) Clone the repository
2) add a .env file and add your Groq API key inside it from https://console.groq.com/keys.   

Inside the .env file add
```
GROQ_API = "your_api_key"
```
3) create a new env from conda. 
```
 conda env create -f environment.yml. 
 ```

4) Run main app. 
```
streamlit run main.py
```


Preview:  

![Alt text](./preview.png)
