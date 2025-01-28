import asyncio
import json
import os
from urllib.parse import unquote
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent
from huggingface_hub import InferenceClient

from csv_service import get_csv_basic_info

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("PYDANTICAI_GROQ_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("PYDANTICAI_GEMINI_API_KEY")


class UserRequirement(BaseModel):
     chat: bool
     chart: bool



action_agent_system_prompt = """
You are an action agent responsible for determining the appropriate response based on a user's query. Your task is to analyze the query and decide whether the required action involves basic calculations, tabular/structured data representation, or generating a true visualization.

**Key Considerations**:
1. **Basic Data Structures**: Requests for outputs like lists, tables, or simple summaries are not considered visualizations and do not require chart generation. These are part of standard data representation.
2. **True Visualizations**: Only queries that explicitly require a graphical representation (e.g., graphs, bar charts, line charts, etc.) should be classified as requiring a visualization.

**Action Categories**:
1. **Calculation-Only Queries**:
   - **Description**: These queries involve computing a value or summarizing data without requiring a visualization.
   - **Example Query**: `What is the total sales for the year 2022?`
   - **Action**:
       - `chat`: True
       - `chart`: False

2. **Queries Requiring Visualization**:
   - **Description**: These queries explicitly request graphical or visual outputs such as trends, comparisons, or data distributions.
   - **Example Query**: `Show a bar chart of sales trends for the last 5 years.`
   - **Action**:
       - `chat`: False
       - `chart`: True

**Output**:
Your response must be accurate and professional, ensuring that visualizations are only marked as required when explicitly requested or genuinely necessary.
"""


     
    
action_agent = Agent(
    #"groq:llama3-70b-8192",
    "gemini-1.5-flash",
    name="Action Agent",
    result_type=UserRequirement,
    system_prompt=action_agent_system_prompt,
)
   
# here we can get groq error
async def isVisualizationQuery(query: str):
 result = await action_agent.run(query)
 return {"chat_required": result.data.chat, "chart_required": result.data.chart} 


##### ----------------------------------------------------------------------------------------------------------------------------------------------- #####

class CasualResponse(BaseModel):
   casual_talk: bool
   response:str

casual_agent_system_prompt = """
 You are an AI that handles both casual conversations and technical conversations only.
 
1. Origin: You are an CSV Chat Assistant bot developed by Soumik Bose and his team and their company's name is "chatcsvandpdf.com"..

2. For casual conversations (like greetings, small talk, etc.), respond in a friendly, informal manner.
   Example queries: "Hi", "Hello", "How are you?", "What's up? etc..."

3. If the query is calculative or requires data processing (like math, analysis, or querying a dataset), do not respond casually. Instead, return a `casual_talk: False` tag and proceed with the necessary computation or processing. This indicates the query requires further handling for calculation, data retrieval, or analysis.
   Example queries: "What is 5 plus 3?", "How much revenue was made in Q4?", "Find the average age from this dataset."

Please ensure that the response is appropriate for the task at hand. For casual talk, maintain a friendly tone. For calculative tasks, process the query accordingly, but include the `casual_talk: False` in the response to signal that the query requires a data-driven approach.
"""

casual_agent = Agent(
    
    "groq:llama3-70b-8192",
    #"gemini-1.5-flash",
    name="Casual_Chat_Agent",
    result_type=CasualResponse,
    system_prompt=casual_agent_system_prompt,
)

async def casual_query(query):
    result = await casual_agent.run(f"{query}")
    return {"casual_talk": result.data.casual_talk, "response": result.data.response}
 
 
##### ----------------------------------------------------------------------------------------------------------------------------------------------- #####


# fallback_agent_system_prompt = '''
# You are the last resort AI assistant developed by Soumik Bose and his team at "chatcsvandpdf.com". All other systems have failed to address the user's query, and you must now provide a solution. Your task is to generate Python code or a response to handle the user's request for a CSV file gracefully. Follow these guidelines:

# 1. **Graceful Handling**: Approach the situation calmly and provide a reliable solution. If information is insufficient, clarify before proceeding.
# 2. **Fallback Expertise**: Advise the user to use appropriate libraries or methods (e.g., pandas for file handling, numpy, seaborn, matplotlib, or similar packages for calculations, analysis, or chart creation) to process, analyze, or manipulate the CSV data effectively.
# 3. **Code and Explanation**: Write clean, modular Python code, and explain key steps with comments for clarity.
# 4. **Summary of CSV**: If applicable, include a small summary of the CSV structure or content for context.
# 5. **Edge Case Management**: Handle edge cases such as:
#    - Empty CSV files
#    - Missing headers or malformed rows
#    - Large datasets
# 6. **User-Friendly Response**: Present the output in a clear and user-friendly manner, alongside the Python code.
# 7. **Example Usage**: Include example input/output or a sample test case to ensure understanding.

# Remember, the user is relying on you as their final hope to solve their CSV-related query. Provide your best effort, and ensure the response is both actionable and informative.
# '''


# class FallbackChatResponse(BaseModel):
#     response: str
    
    
# fallback_agent = Agent(
#     "groq:llama3-70b-8192",
#     name="Fallback_Agent",
#     result_type=FallbackChatResponse,
#     system_prompt=fallback_agent_system_prompt,
# )

# async def fallback_chat(query, csv_url):
#     # Extract the CSV data
#     decoded_url = unquote(csv_url)
   
    
#     # Prepare the string with column names and two rows
#     basic_csv_info = json.dumps(get_csv_basic_info(decoded_url))
    
#     os.environ["GROQ_API_KEY"] = os.getenv("PYDANTICAI_GROQ_FALLBACK_CHAT_API_KEY")
    
#     # Pass the information to the agent
#     result = await fallback_agent.run(f"{query}. basic csv info: {basic_csv_info}")
#     return {"fallback_response": result.data.response}


##### ----------------------------------------------------------------------------------------------------------------------------------------------- #####

hf_api_keys = os.getenv("HF_API_KEYS").split(",")
current_key_index = 0

# Initialize the InferenceClient with the first API key
client = InferenceClient(api_key=hf_api_keys[current_key_index])

async def switch_api_key():
    """
    Switch to the next available API key asynchronously.
    """
    global current_key_index, client
    current_key_index += 1

    if current_key_index < len(hf_api_keys):
        print(f"Switching to the next API key: {hf_api_keys[current_key_index]}")
        client = InferenceClient(api_key=hf_api_keys[current_key_index])
        return True
    else:
        print("No more API keys available.")
        return False


async def fallback_chat(user_query,csv_url):
    """
    Generates a response based on the provided context and user query.
    """
    csv_info = get_csv_basic_info(csv_url)

    system_prompt = '''
        You are the AI assistant developed by Soumik Bose and his team at "chatcsvandpdf.com". All other systems have failed to address the user's query, and you must now provide a solution. Your task is to generate Python code or a response to handle the user's request for a CSV file gracefully. Follow these guidelines:

        1. **Graceful Handling**: Approach the situation calmly and provide a reliable solution. If information is insufficient, clarify before proceeding.
        2. **Fallback Expertise**: Advise the user to use appropriate libraries or methods (e.g., pandas for file handling, numpy, seaborn, matplotlib, or similar packages for calculations, analysis, or chart creation) to process, analyze, or manipulate the CSV data effectively.
        3. **Code and Explanation**: Write clean, modular Python code, and explain key steps with comments for clarity.
        4. **Summary of CSV**: If applicable, include a small summary of the CSV structure or content for context.
        5. **Edge Case Management**: Handle edge cases such as:
           - Empty CSV files
           - Missing headers or malformed rows
           - Large datasets
        6. **User-Friendly Response**: Present the output in a clear and user-friendly manner, alongside the Python code.
        7. **Example Usage**: Include example input/output or a sample test case to ensure understanding.
        
        <basic csv info:>
        {csv_info}
        </basic csv info:>

        Remember, the user is relying on you as their final hope to solve their CSV-related query. Provide your best effort, and ensure the response is both actionable and informative.
        '''
    formatted_system_prompt = system_prompt.format(csv_info=json.dumps(csv_info))

    messages = [
        {"role": "system", "content": formatted_system_prompt},
        {"role": "user", "content": user_query},
    ]

    while current_key_index < len(hf_api_keys):
        try:
            response = await asyncio.to_thread(client.chat.completions.create, 
                                                model="01-ai/Yi-1.5-34B-Chat",
                                                messages=messages,
                                                temperature=0.5,
                                                max_tokens=2048,
                                                top_p=0.7,
                                                stream=False)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with API key {hf_api_keys[current_key_index]}: {e}")
            if not await switch_api_key():
                return "Sorry, we're currently experiencing technical difficulties. Please try again later."



