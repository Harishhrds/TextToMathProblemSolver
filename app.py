import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain,LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import Tool,initialize_agent
from langchain.agents.agent_types import AgentType

st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2")
groq_api_key = st.sidebar.text_input(label="Groq_api_key",type="password")
if not groq_api_key:
    st.info("please enter grqq api key to continue")
    st.stop()
llm = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

# Initializing tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="search the content from the internet and to find answers based on question"
)  

# Initialize math tool
math_chain = LLMMathChain.from_llm(llm=llm) # llmMathChain() sends math-related problems to llm
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answer math related questions.only input math related expression to be provided."
)
prompt = """
your agent for answering users math related questions.Logically arrive at the solution and provide detail explanation
an display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)
# combining all tools
chain = LLMChain(llm=llm,prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="tool used for answer logic based and reasoning questions"

)
## initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role":"assistant","content":"Hi,I'm math a chatbot who can answer all math questions"
        }
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# LEts start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Find Answer"):
    if question:
        with st.spinner("Getting response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message('user').write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True) #st.container()-->Tells Streamlit where to render the step-by-step output.
            #Injects this callback into the LangChain execution so all events (e.g., thoughts, actions, final answers) are streamed live into the Streamlit UI.
            response = assistant_agent.run(st.session_state.messages,callbacks=[st_cb]) 
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("## Response")
            st.success(response)
    else:
        st.warning("please write question")        
    
        

