from agno.tools.youtube import YouTubeTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.groq import Groq
# from agno.models.azure import AzureOpenAI
from agno.tools import tool
from agno.agent import Agent
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

api_key=os.getenv("groq_api_key")
model = Groq(
    id="gemma2-9b-it",
    api_key=api_key,
)

youtube_agent = Agent(
    name = "Youtube agent bot",
    model=model,
    tools=[YouTubeTools()],
    instructions="Give the detailed information of the requested/mentioned youtube video based on it's title/url.",
    role="You are a highly intelligent youtube agent who derives information from youtube videos based on it's title/url.",
    markdown=True,
    show_tool_calls=True,
)

web_agent = Agent(
    name = "Web agent bot",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions="Give the detailed information of the requested query.",
    role="You are a highly intelligent web search agent who derives information from the web.",
    markdown=True,
    show_tool_calls=True,
)

agent = Agent(
    team=[youtube_agent, web_agent],
    model=model,
)

st.title("MultiAI Agent")
st.divider()

query=st.chat_input("Enter the query")


if query:
    st.write(query)
    st.spinner("Generating response....")
    st.divider()
    response = agent.run(query)
    st.write(response.content)

