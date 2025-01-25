# smolagent gemini flash
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import matplotlib
matplotlib.use("Agg")

load_dotenv()

image_file_path = os.getenv("IMAGE_FILE_PATH")
google_api_keys = os.getenv("GOOGLE_API_KEYS").split(",")

# Monkey patch plt.show to save the plot instead of displaying it
def save_plot_instead_of_show():
    """
    Custom function to save the plot instead of displaying it.
    """

    # Generate a unique filename (e.g., using a timestamp)
    filename = image_file_path
    
    # Save the plot to the current working directory
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

# Replace plt.show with the custom function
plt.show = save_plot_instead_of_show



key_index = 0
current_key = google_api_keys[key_index]

def get_model(api_key):
    return LiteLLMModel(
        model_id='gemini/gemini-2.0-flash-exp',
        api_key=api_key
    )

model = get_model(current_key)

csv_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    additional_authorized_imports=["pandas", "numpy", "io", "os", "matplotlib", "seaborn"]
)

system_prompt = f"""
For handling queries, follow these guidelines:

1. **Response Format**:
   - Always return the response in following format:
     - A list of dict.

2. **Calculation-Only Queries**: 
   - Use the libraries: pandas, numpy.
   - To read csv use pd.read_csv(csv_url).

3. **Queries Requiring Visualization**:
   - Use the libraries: pandas, numpy, matplotlib or seaborn (for visualization).
   - Ensure that each value is clearly visible. Adjust the font size, rotate labels, or truncate labels for readability if needed.
   - Do not use plt.show() in the code snippet to display the plot.
   - Save any generated visualizations as `{image_file_path}`.

"""

def csv_smolagent_hf_gemini(file_path, user_query):
    global key_index, current_key, model
    print(f"Running CSV Agent with query: {user_query}")
    
    while key_index < len(google_api_keys):
        try:
            result = csv_agent.run(
                f"{system_prompt} \n\nQuery: {user_query} \n\nfile_path: {file_path}"
            )
            print('Final result:', result)
            if "Error in generating final LLM output" in result:
                raise ValueError("Detected error in LLM output")
            
            if isinstance(result, (float, int)):
                result = str(result)
            return result
        
        except Exception as e:
            print(f"An error occurred: {e}. Switching to next API key...")
            key_index += 1
            if key_index < len(google_api_keys):
                current_key = google_api_keys[key_index]
                model = get_model(current_key)
                csv_agent.model = model
            else:
                print("All API keys have been exhausted.")
                return "Error: Unable to complete the query."

