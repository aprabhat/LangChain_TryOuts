from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | model | output_parser
# As the model used is not aware about the recent events, it will not be able to answer
result = chain.invoke({"input": "tell me something about google gemini"})
print("without RAG:")
print(result)

# RAG
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Gemini_(language_model)")
docs = loader.load()
# we can use this embedding model to ingest documents into a vectorstore
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Now that we have this data indexed in a vectorstore,
# we will create a retrieval chain. This chain will take an incoming question,
# look up relevant documents, then pass those documents along with the original question
# into an LLM and ask it to answer the original question.
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(model, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "tell me something about google gemini?"})
print("\nwith RAG:")
print(response["answer"])
