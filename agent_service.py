import os
from urllib.parse import unquote
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent

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


fallback_agent_system_prompt = """
You are an AI assistant developed by Soumik Bose and his team and their company's name is "chatcsvandpdf.com" that generates Python code based on user requests. Follow these guidelines:
1. Clarify: Ask questions if the request is unclear.
2. Libraries: Use appropriate libraries like `csv`, `pandas`, or `numpy` for handling CSV data.
3. Code Quality: Write clean, efficient, and modular code to process, analyze, or manipulate CSV data.
4. Comments: Add brief comments to explain key parts, especially for data manipulation.
5. Edge Cases: Handle edge cases like empty CSVs, missing headers, or malformed rows.
6. Tests: Provide example usage or unit tests when necessary.
7. Compatibility: Ensure the code works for the specified Python version and always *** use three backticks (```) before and after your code as Markdown ***.
"""


class FallbackChatResponse(BaseModel):
    python_code: str
    
    
fallback_agent = Agent(
    "groq:llama3-70b-8192",
    name="Fallback_Agent",
    result_type=FallbackChatResponse,
    system_prompt=fallback_agent_system_prompt,
)

async def fallback_chat(query, csv_url):
    # Extract the CSV data
    decoded_url = unquote(csv_url)
    df = pd.read_csv(decoded_url)

    # Extract the first two rows and column names
    cols = df.columns.tolist()
    rows = df.head(2).to_json(orient="records")
    
    # Prepare the string with column names and two rows
    csv_structure_info = f"Column Names: {', '.join(cols)}. Example Rows: {rows}"
    
    os.environ["GROQ_API_KEY"] = os.getenv("PYDANTICAI_GROQ_FALLBACK_CHAT_API_KEY")
    
    # Pass the information to the agent
    result = await fallback_agent.run(f"{query}. File Structure: {csv_structure_info}")
    return {"python_code": result.data.python_code}
