import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
from langchain_core.runnables import Runnable
import operator
from typing import Annotated, Sequence, TypedDict , List
from langchain_core.messages import BaseMessage
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, END




def main():

  st.title("LangGraph + Function Call + YahooFinance 👾")
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
            "You are the supervisor over the following agents: {agents}."
            " You are responsible for assigning tasks to each agent as requested by the user."
            " Each agent executes tasks according to their roles and responds with their results and status."
            " Please review the information and answer with the name of the agent to which the task should be assigned next."
            " Answer 'FINISH' if you are satisfied that you have fulfilled the user's request."
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
                "In light of the above conversation, please select one of the following options for which agent should be act or end next: {options}."
              ),
            ]
          ).partial(options=str(options), agents=", ".join(agents))

          return (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="supervisor")
            | JsonOutputFunctionsParser()
          )

        @tool
        def researcher(urls: List[str]) -> str:
          """Use requests and bs4 to scrape the provided web pages for detailed information."""
          loader = WebBaseLoader(urls)
          docs = loader.load()
          return "\n\n".join(
              [
                  f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
                  for doc in docs
              ]
          )


        @tool("Market Analyser")
        def analyze(content: str) -> str:
          """Market Analyser"""
          chat = ChatOpenAI()
          messages = [
          SystemMessage(
            content="You are a market analyst specializing in e-commerce trends, tasked with identifying a winning product to sell on Amazon. "
                    "Your goal is to leverage market analysis datas and your expertise to pinpoint a product that meets specific criteria for "
                    "success in the highly competitive online marketplace "
            ),
            HumanMessage(
              content=content
            ),
          ]
          response = chat(messages)
          return response.content

        @tool("DropShipping_expert")
        def expert(content: str) -> str:
          """Execute a trade"""
          chat = ChatOpenAI()
          messages = [
          SystemMessage(
            content="Act as an experienced DropShopping assistant.Your task is to identify Wining Product"
                    "thorough examination of the product range and pricing"
                    "Provide insights to help decide whether to start selling this product or not"
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
            "You are an amazon scraper."
          )
          return create_agent(llm, [researcher], prompt)

        def analyzer_agent() -> Runnable:
          prompt = (
            "you are analyser data that scrape from amazaon scraper i want you to help to find wining product"
            
          )
          return create_agent(llm, [analyze], prompt)

        def Expert_agent() -> Runnable:
          prompt = (
            "You are a Buyer, your job is to help me decide wthere i start selling product or not"
          )
          return create_agent(llm, [expert], prompt)



        RESEARCHER = "RESEARCHER"
        ANALYZER = "Analyzer"
        EXPERT = "Expet"
        SUPERVISOR = "SUPERVISOR"

        agents = [RESEARCHER , ANALYZER, EXPERT]

        class AgentState(TypedDict):
          messages: Annotated[Sequence[BaseMessage], operator.add]
          next: str

        def scraper_node(state: AgentState) -> dict:
          result = scraper_agent().invoke(state)
          return {"messages": [HumanMessage(content=result["output"], name=RESEARCHER)]}

        def Analyzer_node(state: AgentState) -> dict:
          result = analyzer_agent().invoke(state)
          return {"messages": [HumanMessage(content=result["output"], name=ANALYZER)]}

        def Expert_node(state: AgentState) -> dict:
          result = Expert_agent().invoke(state)
          return {"messages": [HumanMessage(content=result["output"], name=EXPERT)]}

        def supervisor_node(state: AgentState) -> Runnable:
          return create_supervisor(llm, agents)
        

        workflow = StateGraph(AgentState)
        workflow.add_node(RESEARCHER, scraper_node)
        workflow.add_node(ANALYZER,Analyzer_node)
        workflow.add_node(EXPERT,Expert_node)
        workflow.add_node(SUPERVISOR, supervisor_node)

        workflow.add_edge(RESEARCHER, SUPERVISOR)
        workflow.add_edge(ANALYZER,SUPERVISOR)
        workflow.add_edge(EXPERT,SUPERVISOR)
        workflow.add_conditional_edges(
          SUPERVISOR,
          lambda x: x["next"],
          {
            RESEARCHER : RESEARCHER,
            ANALYZER: ANALYZER,
            EXPERT : EXPERT,
            "FINISH" : END
          }
        )

        workflow.set_entry_point(SUPERVISOR)

        graph = workflow.compile()


        for s in graph.stream({"messages": [HumanMessage(content=user_input)]}):
          if "__end__" not in s:
            st.write(s)
            st.write("-----")




        




















        workflow = StateGraph(AgentState)

        workflow.add_node(RESEARCHER, scraper_node)
        workflow.add_node(ANALYZER, Analyzer_node)
        workflow.add_node(EXPERT, Expert_node)
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
            st.write(s)
            st.write("----")

  
if __name__ == "__main__":
    main()

