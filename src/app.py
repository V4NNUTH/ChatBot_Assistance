from dotenv import load_dotenv
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data Science at a CamCyber. You are interacting with a user who is asking you questions about 
    the company's database.
    Base on the table schema below, write a SQL query that would answer the user's question. Take the conversation
    history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and noting else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistID, COUNT(*) as track_count FROM Track GROUP BY ArtistID ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question:{question}
    SQL Query:
    
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    #api_key = os.getenv("OPENAI_API_KEY")
    #llm = ChatOpenAI(model="gpt-4-0125-preview",)
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    # Use the API key in the ChatOpenAI model
    llm = ChatOpenAI(openai_api_key=api_key)

    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db:SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data science at a CamCyber. You are interacting with a user who is asking you question about the 
    company's database.
    Base on the table schema below, question, sql query, and sql response, write a natural language response as Khmer Lenguage.
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key="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")
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema = lambda _: db.get_table_info(),
            response = lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    }
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="សួស្តី‌!ខ្ញុំគឺជំនួយការCamCyberBot. តើអ្នកមានសំណួរអ្វីចង់សួរទាក់ទង់ហ្នឹងពត៍មាននៅក្នុងDataBase?"),
    ]

load_dotenv()

st.set_page_config(page_title="CamCyber Bot",page_icon=":speech_balloon:")

#st.title("CamCyber Bot")
st.markdown(
    """
    <h1 style='color: lightblue; text-align: center;'>CamCyber Bot</h1>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.subheader("Settings")
    st.write("នេះជាជំនួយការរបស់CamCyber.សូមភ្ជាប់ជាមួយDataBase ដើម្បីសួរពត៍មាន.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", key="Password")
    st.text_input("DataBase", value="Chinook", key="Database")
    
    if st.button("Connect"):
        with st.spinner("កំពុងភ្ជាប់ទៅកាន់ Database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("ការភ្ជាប់ជោគជ័យ!")


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    
user_query = st.chat_input("តើអ្នកមានអ្វីដែលអាចឱ្យខ្ញុំជួយបាន....")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    # with st.chat_message("AI"):
    #     sql_chain = get_sql_chain(st.session_state.db)
    #     response = sql_chain.invoke({
    #         "chat_history": st.session_state.chat_history,
    #         "question": user_query
    #     })
    #     #response = "I don't know how to respond to that."
    #     st.markdown(response)
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))