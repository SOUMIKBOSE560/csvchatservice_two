import os
import time
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
import matplotlib

matplotlib.use("Agg")

# Load environment variables
load_dotenv()
hf_tokens = os.getenv("HF_TOKENS").split(",")
image_file_path = os.getenv("IMAGE_FILE_PATH")

# Set the first HF token globally
os.environ["HF_TOKEN"] = hf_tokens[0]
print(f"Initial HF_TOKEN set to the first token.")

def initialize_agent():
    """Reinitialize the agent to pick up the latest HF_TOKEN."""
    return CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=HfApiModel(),
        additional_authorized_imports=["pandas", "numpy", "io", "matplotlib", "seaborn"]
    )

# Initialize agent
csv_agent = initialize_agent()

system_prompt = f"""
For handling queries, follow these guidelines:

1. **Calculation-Only Queries**: 
   - Use the libraries: pandas, numpy.
   - To read csv use pd.read_csv(csv_url).

2. **Queries Requiring Visualization**:
   - Use the libraries: pandas, numpy, matplotlib or seaborn (for visualization).
   - Please ensure that each value is clearly visible. Adjust the font size, rotate labels, or truncate labels for readability if needed.
   - Do not use plt.show() in the code snippet to display the plot.
   - Save any generated visualizations as `{image_file_path}`.
"""

def switch_global_hf_token(current_index):
    """Switch the global HF_TOKEN environment variable and reinitialize the agent."""
    next_index = current_index + 1
    if next_index < len(hf_tokens):
        os.environ["HF_TOKEN"] = hf_tokens[next_index]
        print(f"Switched globally to HF_TOKEN #{next_index + 1}")
        return next_index
    else:
        print("All HF tokens exhausted.")
        return -1  # Indicate no more tokens available

def csv_smolagent_hf_qwen(file_path, user_query):
    global csv_agent
    print(f"Running CSV Agent with query: {user_query}")

    token_index = 0
    while token_index != -1:
        try:
            # Ensure agent uses the current HF_TOKEN
            csv_agent = initialize_agent()

            # Log the current HF_TOKEN being used
            print(f"Using HF_TOKEN: {os.environ['HF_TOKEN']}")

            # Run agent query
            result = csv_agent.run(
                f"{system_prompt} \n\nQuery: {user_query} \n\nfile_path: {file_path}"
            )

            # Log the result
            print('Final result:', result)

            # Check for specific error in result
            if "Error in generating final LLM output" in result:
                print("Encountered error in LLM output. Switching token globally...")
                token_index = switch_global_hf_token(token_index)
                time.sleep(2)  # Add delay before retrying
                continue

            # Convert numeric results to string
            if isinstance(result, (float, int)):
                result = str(result)
            return result  # Success, exit loop

        except Exception as e:
            print(f"Error: {e}. Switching to next global API key...")
            token_index = switch_global_hf_token(token_index)
            
    print("All API keys exhausted.")
    return "Error: Unable to complete the query."
