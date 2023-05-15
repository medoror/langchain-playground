from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
import pinecone
from langchain.vectorstores import Pinecone

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

load_dotenv()


## Playground from video: https://www.youtube.com/watch?v=aywZrzNaKjs

if __name__ == '__main__':
    llm = OpenAI(model_name="text-davinci-003")
    print(llm("explain large language models in one sentence"))

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    messages = [
        SystemMessage(content="You are an expert data scientist"),
        HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
    ]

    response = chat(messages)
    print(response.content, end='\n')

    template = """
    You are an expert data scientist with an expertise in building deep learning models.
    Explain the concept of {concept} in a couple of lines
    """

    prompt = PromptTemplate(
        input_variables=["concept"],
        template=template
    )

    llm(prompt.format(concept="autoencoder"))

    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run("autoencoder"))

    second_prompt = PromptTemplate(
        input_variables=["ml_concept"],
        template="Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words",
    )

    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

    explanation = overall_chain.run("autoencoder")
    print(explanation)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0
    )

    texts = text_splitter.create_documents([explanation])

    embeddings = OpenAIEmbeddings(model="ada")

    query_result = embeddings.embed_query(texts[0].page_content)
    print(query_result)

    # seems to be too big for the default pinecone free service

    # pinecone.init(
    #     api_key=os.getenv('PINECONE_API_KEY'),
    #     environment=os.getenv('PINECONE_ENV')
    # )
    #
    # index_name = "langchain-quickstart"
    #
    # pinecone.create_index(index_name, dimension=1536,
    #                       metric="cosine", pods=1, pod_type="p1.x1")
    #
    # search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    #
    # query = "What is magical about an autoencoder?"
    #
    # result = search.similarity_search(query)
    #
    # print(result)

    agent_executor = create_python_agent(
        llm=OpenAI(temperature=0, max_tokens=1000),
        tool=PythonREPLTool(),
        verbose=True
    )

    agent_executor.run("Find the roots (zeros) if the quadtratic function 3 * x**2 + 2*x -1")

