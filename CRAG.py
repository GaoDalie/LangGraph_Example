from dotenv import load_dotenv
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from langchain.prompts import PromptTemplate
import pprint
import os
# Load environment variables
load_dotenv()


run_local = 'No'
models = "openai"
openai_api_key = "Your_API_KEY"
google_api_key = "Your_API_KEY"
local_llm = 'Solar'
os.environ["TAVILY_API_KEY"] = ""
        

# Split documents
url  = 'https://lilianweng.github.io/posts/2023-06-23-agent/'
loader = WebBaseLoader(url)
docs = loader.load()


# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
all_splits = text_splitter.split_documents(docs)


# Embed and index
if run_local == 'Yes':
    embeddings = GPT4AllEmbeddings()
elif models == 'openai':
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
else:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

# Index
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()
print(retriever)

###################################################################


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]
#############################################################


### Nodes ###

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    local = state_dict["local"]
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "local": local, "question": question}}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM Setup
    if run_local == "Yes":
        llm = ChatOllama(model=local_llm, 
                        temperature=0)
    elif models == "openai" :
        llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            temperature=0 , 
            openai_api_key=openai_api_key
        )
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                    google_api_key=google_api_key,
                                    convert_system_message_to_human = True,
                                    verbose = True,
        )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # LLM
    if run_local == "Yes":
        llm = ChatOllama(model=local_llm, 
                        temperature=0)
    elif models == "openai" :
        llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            temperature=0 , 
            openai_api_key=openai_api_key
        )
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                    google_api_key=google_api_key,
                                    convert_system_message_to_human = True,
                                    verbose = True,
        )
    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        score: str = Field(description="Relevance score 'yes' or 'no'")

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=grade)

    from langchain_core.output_parsers import JsonOutputParser

    parser = JsonOutputParser(pydantic_object=grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with no premable or explaination and use these instructons to format the output: {format_instructions}""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": d.page_content,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform web search
            continue

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "local": local,
            "run_web_search": search,
        }
    }
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any premable, only respond with the updated question: """,
        input_variables=["question"],
    )

    # Grader
    # LLM
    if run_local == "Yes":
        llm = ChatOllama(model=local_llm, 
                        temperature=0)
    elif models == "openai" :
        llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            temperature=0 , 
            openai_api_key=openai_api_key
        )
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                    google_api_key=google_api_key,
                                    convert_system_message_to_human = True,
                                    verbose = True,
        )
    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "keys": {"documents": documents, "question": better_question, "local": local}
    }

def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """

    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]
    try:
        tool = TavilySearchResults()
        docs = tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
    except Exception as error:
        print(error)

    return {"keys": {"documents": documents, "local": local, "question": question}}

### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Run
inputs = {
    "keys": {
        "question": 'Explain how the different types of agent memory work?',
        "local": run_local,
    }
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        print(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")

# Final generation
pprint.pprint(value['keys']['generation'])

import os
import requests
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
from langchain_core.runnables import Runnable
import operator
import streamlit as st


def main():

  
  st.title("LangGraph + Function Call + Amazaon Scraper 👾")
  # Add a sidebar for model selection
  OPENAI_MODEL = st.sidebar.selectbox(
        "Select Model",
        ["gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]  # Add your model options here
    )
  
  api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

  if api_key:
   os.environ["OPENAI_API_KEY"] = api_key

  user_input = st.text_input("Enter your input here:")

  # Run the workflow
  if st.button("Run Workflow"):
      with st.spinner("Running Workflow..."):

        def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_openai_tools_agent(llm, tools, prompt)
            return AgentExecutor(agent=agent, tools=tools)

        def create_supervisor(llm: ChatOpenAI, agents: list[str]):
            system_prompt = (
                f"You are the supervisor over the following agents: {', '.join(agents)}. "
                "You are responsible for assigning tasks to each agent as requested by the user. "
                "Each agent executes tasks according to their roles and responds with their results and status. "
                "Please review the information and answer with the name of the agent to which the task should be assigned next. "
                "Answer 'FINISH' if you are satisfied that you have fulfilled the user's request."
            )

            options = ["FINISH"] + agents
            function_def = {
                "name": "supervisor",
                "description": "Select the next agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "next": {
                            "anyOf": [
                                {"enum": options},
                            ],
                        }
                    },
                    "required": ["next"],
                },
            }

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                    (
                        "system",
                        "In light of the above conversation, please select one of the following options for which agent should act or end next: {options}."
                    ),
                ]
            ).partial(options=str(options), agents=", ".join(agents))

            return (
                prompt
                | llm.bind_functions(functions=[function_def], function_call="supervisor")
                | JsonOutputFunctionsParser()
            )

        def researcher(query):
            
            """
            Scrape product titles and prices from the given Amazon URL.
            """
            url = f"https://www.amazon.com/s?k={query}"
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}
            page = requests.get(url, headers=headers)
            soup = BeautifulSoup(page.content, "html.parser")

            product_details = []

            # Find all the product containers
            product_containers = soup.find_all("div", {"data-component-type": "s-search-result"})

            for product in product_containers:
                product_info = {}

                # Obtain title of the product
                title = product.find("span", {"class": "a-size-medium"})
                if title:
                    product_info['title'] = title.get_text(strip=True)

                # Obtain price of the product
                price = product.find("span", {"class": "a-price-whole"})
                if price:
                    product_info['price'] = price.get_text(strip=True)

                # Add more details as needed

                product_details.append(product_info)

            return product_details

        @tool("Amazon_Research")
        def researcher_tool(query: str) -> str:
            """Research by Scraper"""
            func = lambda word: researcher(word)
          
            return func

        @tool("Market Analyser")
        def analyze_tool(content: str) -> str:
            """Market Analyser"""
            chat = ChatOpenAI()
            messages = [
                SystemMessage(
                    content="You are a market analyst specializing in e-commerce trends, tasked with identifying a winning product to sell on Amazon. "
                            "Your goal is to leverage market analysis data and your expertise to pinpoint a product that meets specific criteria for "
                            "success in the highly competitive online marketplace "
                ),
                HumanMessage(
                    content=content
                ),
            ]
            response = chat(messages)
            return response.content

        @tool("DropShipping_expert")
        def expert_tool(content: str) -> str:
            """Execute a trade"""
            chat = ChatOpenAI()
            messages = [
                SystemMessage(
                    content="Act as an experienced DropShipping assistant. Your task is to identify Winning Product "
                            "through examination of the product range and pricing. "
                            "Provide insights to help decide whether to start selling this product or not."
                ),
                HumanMessage(
                    content=content
                ),
            ]
            response = chat(messages)
            return response.content

        llm = ChatOpenAI(model=OPENAI_MODEL)

        def scraper_agent() -> Runnable:
            prompt = (
                "You are an Amazon scraper."
            )
            return create_agent(llm, [researcher_tool], prompt)

        def analyzer_agent() -> Runnable:
            prompt = (
                "You are analyzing data scraped from Amazon. I want you to help find a winning product."
            )
            return create_agent(llm, [analyze_tool], prompt)

        def expert_agent() -> Runnable:
            prompt = (
                "You are a buyer. Your job is to help me decide whether to start selling a product or not."
            )
            return create_agent(llm, [expert_tool], prompt)

        RESEARCHER = "RESEARCHER"
        ANALYZER = "Analyzer"
        EXPERT = "Expert"
        SUPERVISOR = "SUPERVISOR"

        agents = [RESEARCHER, ANALYZER, EXPERT]

        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]
            next: str

        def scraper_node(state: AgentState) -> dict:
            result = scraper_agent().invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name=RESEARCHER)]}

        def analyzer_node(state: AgentState) -> dict:
            result = analyzer_agent().invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name=ANALYZER)]}

        def expert_node(state: AgentState) -> dict:
            result = expert_agent().invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name=EXPERT)]}

        def supervisor_node(state: AgentState) -> Runnable:
            return create_supervisor(llm, agents)

        workflow = StateGraph(AgentState)

        workflow.add_node(RESEARCHER, scraper_node)
        workflow.add_node(ANALYZER, analyzer_node)
        workflow.add_node(EXPERT, expert_node)
        workflow.add_node(SUPERVISOR, supervisor_node)

        workflow.add_edge(RESEARCHER, SUPERVISOR)
        workflow.add_edge(ANALYZER, SUPERVISOR)
        workflow.add_edge(EXPERT, SUPERVISOR)
        workflow.add_conditional_edges(
          SUPERVISOR,
          lambda x: x["next"],
          {
            RESEARCHER : RESEARCHER ,
            ANALYZER: ANALYZER,
            EXPERT: EXPERT,
            "FINISH": END
          }
        )

        workflow.set_entry_point(SUPERVISOR)

        graph = workflow.compile()

        #what are some of the most popular stocks for 2024 should i invest in or stock that might have the biggest gains in the future
        #What are some of the stocks that had the greatest performance recently And are also the most liquid and highly traded ?
        # User_input = (
        #   "sCRAPE THIS  game laptop and help me to find wining product"
        # )

        for s in graph.stream({"messages": [HumanMessage(content=user_input)]}):
          if "__end__" not in s:
            print(s)
            print("----")

  
if __name__ == "__main__":
    main()

