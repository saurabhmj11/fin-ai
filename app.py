import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import io
import sys
import re

# Load environment variables
load_dotenv()

# Initialize agents
agent_search = Agent(
    name="web search agent",
    role="search web for information",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

agent_finance = Agent(
    name="finance agent",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display data."],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[agent_search, agent_finance],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Function to clean up terminal output
def clean_terminal_output(output):
    # Remove ANSI escape codes using regex
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', output)

# Streamlit UI
st.title("FIN-AI Agent Web App")

# User Input
query = st.text_input("Enter your query:", "")

if st.button("Submit Query"):
    if query.strip():
        with st.spinner("Processing your request..."):
            # Capture terminal output
            output_buffer = io.StringIO()
            sys.stdout = output_buffer  # Redirect stdout
            try:
                multi_ai_agent.print_response(query)  # Process the query
                sys.stdout = sys.__stdout__  # Restore stdout
                raw_output = output_buffer.getvalue()  # Get raw output
                cleaned_output = clean_terminal_output(raw_output)  # Clean it
                if cleaned_output.strip():
                    st.markdown("### Response:")
                    st.markdown(f"```\n{cleaned_output}\n```")  # Display as code block
                else:
                    st.warning("No response was generated.")
            except Exception as e:
                sys.stdout = sys.__stdout__  # Restore stdout in case of errors
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a valid query!")
