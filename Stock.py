import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage




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
        

        @tool("Trading_Research")
        def researcher(query: str) -> str:
          """Research by Yahoo"""
          Yfinance = YahooFinanceNewsTool()
          return Yfinance.run(query)

        @tool("Market Analysist")
        def analyze(content: str) -> str:
          """Market Analyser"""
          chat = ChatOpenAI()
          messages = [
          SystemMessage(
            content="Act as a day trading assistant. Your task is to identify trading assets that meet the specified{User_input}"
                    "Utilize your expertise and available market analysis tools to scan, filter, and evaluate potential assets for trading." 
                    "Once identified, create a comprehensive list with supporting data for each asset, indicating why it meets the criteria. "
                    "Ensure that all information is up-to-date and relevant to the current market conditions. "
            ),
            HumanMessage(
              content=content
            ),
          ]
          response = chat(messages)
          return response.content

        @tool("Trade Execution")
        def executer(content: str) -> str:
          """Execute a trade"""
          chat = ChatOpenAI()
          messages = [
          SystemMessage(
            content="Act as an experienced trading assistant. Based on your comprehensive analysis of current market conditions,"
                    "historical data, and emerging trends, decide on optimal entry, stop-loss, and target points for a specified "
                    "trading asset. Begin by thoroughly reviewing recent price action, key technical indicators, and relevant news"
                    "that might influence the asset's direction."
            ),
            HumanMessage(
              content=content
            ),
          ]
          response = chat(messages)
          return response.content


        from langchain_core.runnables import Runnable

        llm = ChatOpenAI(model=OPENAI_MODEL)

        def researcher_agent() -> Runnable:
          prompt = (
            "You are an Trader research assistant, you uses Yahoo Fiance News to find the most up-to-date and correct information."
            "Your research should be rigorous, data-driven, and well-documented"
          )
          return create_agent(llm, [researcher], prompt)

        def analyzer_agent() -> Runnable:
          prompt = (
            "As a Market Stock Analyzer, your main job is to study the stock market and "
            "help people make smart decisions about their investments "
          )
          return create_agent(llm, [analyze], prompt)

        def executor_agent() -> Runnable:
          prompt = (
            "You are a Executor in the stock market, your job is to help people invest their money wisely."
            "You study how the stock market works and figure out which companies are good to invest in."
          )
          return create_agent(llm, [analyze], prompt)
  


        RESEARCHER = "RESEARCHER"
        ANALYZER = "Analyzer"
        EXECUTOR = "Executor"
        SUPERVISOR = "SUPERVISOR"

        agents = [RESEARCHER, ANALYZER, EXECUTOR]

        class AgentState(TypedDict):
          messages: Annotated[Sequence[BaseMessage], operator.add]
          next: str

        def researcher_node(state: AgentState) -> dict:
          result = researcher_agent().invoke(state)
          return {"messages": [HumanMessage(content=result["output"], name=RESEARCHER)]}
        
        def Analyzer_node(state: AgentState) -> dict:
          result = Analyzer_node().invoke(state)
          return {"messages": [HumanMessage(content=result['output'], name=ANALYZER)]}
        
        def Executor_node(state: AgentState) -> dict: 
          result = executor_agent().invoke(state)
          return {"messages":[HumanMessage(content=result["output"], name=EXECUTOR)]}
        
        def supervisor_node(state: AgentState) -> Runnable:
          return create_supervisor(llm,agents)
        

        workflow = StateGraph(AgentState)

        workflow.add_node(RESEARCHER,researcher_node)
        workflow.add_node(ANALYZER,Analyzer_node)
        workflow.add_node(EXECUTOR, Executor_node)
        workflow.add_node(SUPERVISOR, supervisor_node)

        workflow.add_edge(RESEARCHER, SUPERVISOR)
        workflow.add_edge(ANALYZER, SUPERVISOR)
        workflow.add_edge(EXECUTOR,SUPERVISOR)

        workflow.add_conditional_edges(
          SUPERVISOR,
          lambda x: x["next"],
          {
            RESEARCHER: RESEARCHER,
            ANALYZER: ANALYZER,
            EXECUTOR : EXECUTOR,
            "FINISH" : END
          }
        )

        workflow.set_entry_point(SUPERVISOR)

        graph = workflow.compile()

        for s in graph.stream({"messages" : [HumanMessage(content=user_input)]}):
          if "__end__" not in s:
            st.write(s)
            st.write("----")




        






















        def Analyzer_node(state: AgentState) -> dict:
          result = analyzer_agent().invoke(state)
          return {"messages": [HumanMessage(content=result["output"], name=ANALYZER)]}

        def Executor_node(state: AgentState) -> dict:
          result = executor_agent().invoke(state)
          return {"messages": [HumanMessage(content=result["output"], name=EXECUTOR)]}

        def supervisor_node(state: AgentState) -> Runnable:
          return create_supervisor(llm, agents)

        from langgraph.graph import StateGraph, END

        workflow = StateGraph(AgentState)

        workflow.add_node(RESEARCHER, researcher_node)
        workflow.add_node(ANALYZER, Analyzer_node)
        workflow.add_node(EXECUTOR, Executor_node)
        workflow.add_node(SUPERVISOR, supervisor_node)

        workflow.add_edge(RESEARCHER, SUPERVISOR)
        workflow.add_edge(ANALYZER, SUPERVISOR)
        workflow.add_edge(EXECUTOR, SUPERVISOR)
        workflow.add_conditional_edges(
          SUPERVISOR,
          lambda x: x["next"],
          {
            RESEARCHER: RESEARCHER,
            ANALYZER: ANALYZER,
            EXECUTOR: EXECUTOR,
            "FINISH": END
          }
        )

        workflow.set_entry_point(SUPERVISOR)

        graph = workflow.compile()

        #what are some of the most popular stocks for 2024 should i invest in or stock that might have the biggest gains in the future
        #What are some of the stocks that had the greatest performance recently And are also the most liquid and highly traded ?
        # user_input = (
        #   "what are some of the most popular stocks for 2023 should i invest in or stock that might have the biggest gains in the future"
        # )

        for s in graph.stream({"messages": [HumanMessage(content=user_input)]}):
          if "__end__" not in s:
            st.write(s)
            st.write("-----")


if __name__ == "__main__":
    main()