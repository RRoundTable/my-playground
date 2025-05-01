from typing import Annotated, TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the state type
class AgentState(TypedDict):
    title: str
    content: str
    evaluation: str
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7
)

# Define the evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert youtube video title evaluator. Your task is to evaluate how well the given title matches the content and the target audience.
    The target audience is Korean language learners who are passionate about Korean culture, including K-pop, K-dramas, Korean cuisine, beauty trends, fashion, and overall cultural aspects.
    
    Consider the following aspects:
    1. Simplicity: The title should be simple so viewers can understand it at a glance.
    2. Unexpectedness: Titles that break viewers' consistent expectations are more likely to attract attention.
    3. Specificity: Titles that are too difficult interfere with viewers' reading. Use easy and specific language considering the target audience.
    4. Credibility: If there are institutions or individuals that can add credibility to the title, including them can make viewers trust it more.
    5. Storytelling: People remember stories more easily.
    6. Emotion: Emotional titles evoke feelings in viewers and are more memorable.
     
    A good title needs to satisfy at least one of the above conditions. And it doesn't need to satisfy all of them.
     
    And please write your response in Korean.
    
    Provide a detailed evaluation and suggest improvements if needed."""),
    ("system", """좋은 제목 예시:
    1. 한국어를 읽는 가장 쉬운 방법: 이 제목은 단순해서 쉽게 이해할 수 있어
    2. 1400회 수업을 한 프로가 알려주는 한국어 공부 방법: 1400회라는 횟수는 시청자에게 신뢰감을 줄 수 있어
    3. 여행할때 자주쓰는 한국어 표현 10개: 구체적으로 여행상황에서 사용하는 표현과 개수를 명시해서 시청자가 내용을 명확하게 이해할 수 있어
    4. 한국어 21일만에 배우는 방법: 21일이라는 기간은 시청자에게 구체적인 목표설정에 도움을 줄 수 있어.
    
    """),
    ("human", """
    Title: {title}
    
    Content: {content}
    
    Please evaluate this title and provide your analysis.
    """)
])

# Define the chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant specialized in Korean language learning and YouTube content creation.
    You can help users evaluate their video titles and provide suggestions for improvement.
    You can also engage in general conversation about Korean language learning and content creation.
    Please respond in Korean.

    대화를 진행하면서 제목짓는 것을 도와줘 
    
    """),

    ("human", """
    {messages}
    """)
])

# Define the evaluation node
def evaluate_title(state: AgentState) -> AgentState:
    # Generate evaluation
    chain = evaluation_prompt | llm
    response = chain.invoke({
        "title": state["title"],
        "content": state["content"]
    })
    
    # Update state with evaluation
    state["evaluation"] = response.content
    return state

# Define the chat node
def chat(state: AgentState) -> AgentState:
    # Generate response
    chain = chat_prompt | llm
    print("state", state["messages"])
    response = chain.invoke({
        "messages": state["messages"]
    })
    
    # Add AI response to messages
    state["messages"].append(AIMessage(content=response.content))
    return state

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("evaluate", evaluate_title)
workflow.add_node("chat", chat)

# Add edges
workflow.add_edge(START, "evaluate")
workflow.add_edge("evaluate", END)

# Compile the graph
app = workflow.compile()

def evaluate_title_agent(title: str, content: str) -> str:
    """
    Evaluate a title based on its content using the agent.
    
    Args:
        title (str): The title to evaluate
        content (str): The content to compare against
        
    Returns:
        str: The evaluation result
    """
    result = app.invoke({
        "title": title,
        "content": content,
        "evaluation": "",
        "messages": []
    })
    return result["evaluation"]

def chat_with_agent(message: str, history: List[Union[HumanMessage, AIMessage, SystemMessage]] = None) -> str:
    """
    Chat with the title agent.
    
    Args:
        message (str): The user's message
        history (List[Union[HumanMessage, AIMessage, SystemMessage]], optional): Chat history
        
    Returns:
        str: The agent's response
    """
    if history is None:
        history = []
    
    history.append(HumanMessage(content=message))
    # Create state with existing history
    state = {
        "title": "",
        "content": "",
        "evaluation": "",
        "messages": history.copy()  # Create a copy to avoid modifying the original
    }
    
    # Run chat
    result = chat(state)
    return result["messages"][-1].content

# Test section
if __name__ == "__main__":
    # Test evaluation
    # test_title = "한국어 학습의 효과적인 방법"
    # test_content = """
    # 한국어를 배우는 외국인들을 위한 효과적인 학습 방법을 소개합니다.
    # 듣기, 말하기, 읽기, 쓰기의 균형 잡힌 학습이 중요하며,
    # 실제 한국인과의 대화 기회를 많이 가지는 것이 도움이 됩니다.
    # 또한 K-pop과 한국 드라마를 통한 문화 학습도 언어 습득에 큰 도움이 됩니다.
    # 체계적인 문법 학습과 함께 실생활에서 사용되는 표현을 익히는 것이 핵심입니다.
    # """
    
    # print("Test Case 1 - Evaluation:")
    # print("Title:", test_title)
    # print("\nContent:", test_content)
    # print("\nEvaluation:")
    # print(evaluate_title_agent(test_title, test_content))
    
    # Test chat
    print("\nInteractive Chat Session:")
    print("Type 'quit' to exit")
    
    history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        response = chat_with_agent(user_input, history)
        print("Agent:", response)
        
        # Update history with both user input and agent response
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))