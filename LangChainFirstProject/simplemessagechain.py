from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

parser = StrOutputParser()

# Chain (model → parser)
chain = model | parser

messages = [
    SystemMessage(content="Translate the following from English into Italian."),
    HumanMessage(content="hi!"),
]

if __name__ == "__main__":
    print(chain.invoke(messages))
