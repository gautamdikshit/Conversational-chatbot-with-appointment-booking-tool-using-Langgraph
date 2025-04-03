import streamlit as st
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from typing import Dict
import json
from pathlib import Path
import os

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

@tool
def booking_appointment(details: Dict[str, str]) -> Dict[str, str]:
    """
    Books an appointment using the collected details.
    Saves the appointment to a local JSON file.
    """
    file_path = Path("appointments.json")
    appointments = json.loads(file_path.read_text()) if file_path.exists() else []
    appointments.append(details)
    file_path.write_text(json.dumps(appointments, indent=4))
    
    confirmation_message = (f"Appointment booked successfully!\n"
                            f"Name: {details['name']}\n"
                            f"Email: {details['email']}\n"
                            f"Date: {details['date']}\n"
                            f"Time: {details['time']}")
    return {"message": confirmation_message}

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192")
llm_with_tools = llm.bind_tools([booking_appointment])

system_message = SystemMessage(content="""
You are an intelligent assistant that makes normal conversation with the user and also helps them book appointments.
Follow these steps:
1. Chat normally until the user asks to book an appointment.
2. Collect their name, email, date, and time step by step.
3. Confirm the details before booking.
4. Use the booking_appointment tool to finalize the appointment.
""")

def assistant(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke([system_message] + messages)
    return {"messages": [response]}

# Define Graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([booking_appointment]))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

react_graph = builder.compile()

# Streamlit UI
st.title("Conversational Chatbot with Appointment Booking Feature")
st.write("Chat with the assistant below:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

user_input = st.chat_input("Type your message...")

if user_input:
    # Append and display the user message immediately
    user_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(user_message)

    # Force UI refresh before processing response
    st.rerun()

# Process response **only if the last message was from the user**
if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
    response = react_graph.invoke({"messages": st.session_state.chat_history})
    last_message = response["messages"][-1]
    
    # Append AI response
    st.session_state.chat_history.append(last_message)

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(last_message.content)

    # Force another UI refresh to show updated messages
    st.rerun()
