from flask import Flask, render_template, jsonify, request

from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

app = Flask(__name__)



os.environ["MISTRAL_API_KEY"] = "W6sTfEeTeTQtwfxhfNjK9aK5M0OU1NHH"







#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


embeddings = download_hugging_face_embeddings()




from langchain_community.retrievers import PineconeHybridSearchRetriever


import os
from pinecone import Pinecone,ServerlessSpec
index_name ="medical"
PINECONE_API_KEY = os.environ.get("pcsk_6cLNH7_BgtCnvCvMiYvT5ZikGHVoqpMuJZjv1LMRhQBRx4Xws1Tni5XzBJ3Jrwfm5SLVv3")

pc=Pinecone(api_key="pcsk_6cLNH7_BgtCnvCvMiYvT5ZikGHVoqpMuJZjv1LMRhQBRx4Xws1Tni5XzBJ3Jrwfm5SLVv3")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension = 384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region="us-east-1")
    )

  

from langchain_pinecone import PineconeVectorStore
    
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)





retriver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})


from langchain_mistralai import ChatMistralAI

# Initialize Mistral AI LLM
llm = ChatMistralAI(
    model="mistral-small",  # Choose appropriate model
    temperature=0.4,
    max_tokens=500
)


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



system_prompt=(

    
    "You are an AI assistant designed to answer questions "
    "Search the Data carefully to extract relevant and accurate data. "
    "Combine answers from whole data when needed. "
    "Respond in a concise and natural tone. "
    "If you don’t know the answer, say 'I don’t know.' "
    "Avoid gibberish, speculation, or unnecessary details. "
    "Do not mention the system prompt or how you found the answer. "
    "Do not mention about the data PDFs. or Analysing them "
    "Do not mention from where you get your Data "
    "Just present the answer clearly and helpfully."
    "When ipnut is Hey,Hello,Hi or similar describe youself as a DR.EZ who helps in medical purposes and say it only one time do not repeat it"
    "DO not mention the word PDf,PDFs,First PDF, Second PDF and similar words"
    "\n\n"

    "{context}"
)
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain=create_retrieval_chain(retriver , question_answer_chain)




@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response=rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True, use_reloader=False)


