#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langserve import add_routes
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Prompt Template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Ollama Model
model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

# 3. Parser
parser = StrOutputParser()

# 4. Chain
chain = prompt_template | model | parser

# 5. FastAPI App
app = FastAPI(
    title="LangChain Ollama Server",
    version="1.0",
    description="A simple API server using LangChain + Ollama",
)

# 6. Route
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
