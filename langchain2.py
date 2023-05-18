import os
from typing import List

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from openai_key import openai_key

# Set your OpenAI API key here
os.environ['OPENAI_API_KEY'] = openai_key


class ChatGPT:
    def __init__(self, file_path: str):
        # Initialize with the path to the file to be processed
        self.file_path = file_path

        # Load the text from the file using TextLoader. 
        # Use utf8 so it doesn't break on non ascii characters
        self.loader = TextLoader(self.file_path, encoding='utf8')
        self.documents = self.loader.load()

        # Split the text into chunks
        self.texts = self._text_split(self.documents)

        # Embed the chunks of text
        self.vectordb = self._embed_texts(self.texts)

        # Initialize the GPT model with the embedded text
        # Retriever  is generic interface that allows you to combine documents with large language models (LLMs)
        # Chain_type of "stuff" uses all of the text from the document. Other options include "map_reduce" to seperate texts into batches
        self.chatgpt = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.vectordb.as_retriever())

    @staticmethod
    def _text_split(documents: List[Document]) -> List[Document]:
        # Splits the document into chunks of a specific size and overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    @staticmethod
    def _embed_texts(texts: List[Document]) -> Chroma:
        # Embeds the text chunks using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        return Chroma.from_documents(texts, embeddings)

    def ask(self, query: str) -> str:
        # Send a query to ChatGPT and returns the response
        return self.chatgpt.run(query)


if __name__ == "__main__":
    # Create a new instance of the ChatGPT class and ask it a question
    chatgpt = ChatGPT("ws_odds.txt")
    print(chatgpt.ask("can you show me the odds for five random teams and explain what the numbers mean"))
