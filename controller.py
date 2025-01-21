
from urllib.parse import unquote
from fastapi import FastAPI, HTTPException, Header
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from initial_question_handler import if_initial_chart_question, if_initial_chat_question
from smolagent_hf_gemini import csv_smolagent_hf_gemini as gemini_agent
from smolagent_hf_default_qwen import csv_smolagent_hf_qwen as qwen_agent
from agent_service import casual_query, fallback_chat, isVisualizationQuery




# uvicorn controller:app --host localhost --port 8086 --reload
app = FastAPI()
image_file_path = os.getenv("IMAGE_FILE_PATH")
image_not_found = os.getenv("IMAGE_NOT_FOUND")
allowed_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    #allow_origins=allowed_hosts,  # Allows all origins
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)


@app.get("/ping")
async def root():
    return {"message": "Pong !!"}



# Define the request body schema
class InputRequest(BaseModel):
    query: str

@app.post("/api/if_chart")
async def if_chart(query: InputRequest, authorization: str = Header(None)):
    # Check if Authorization header is provided or not
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # Check if Authorization header has the correct format
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    # Extract token from the Authorization header
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")

    # Validate token first
    if token != os.getenv("AUTH_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid token")
    
    # Process the query and CSV URL
    query = query.query

    # Example of processing the query
    response = await isVisualizationQuery(query)
    # print(f"Response if chart: {response}")

    # Return the response
    return {"message": response.get("chart_required")}



@app.post("/api/csv-chat/")
async def csv_chat(request: dict, authorization: str = Header(None)):

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")

    if token != os.getenv("AUTH_TOKEN"):  
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        query = request.get("query")
        query = query.lower()
        
        csv_url = request.get("csv_url")
        decoded_url = unquote(csv_url)
        
            
        # initial query handler
        if(if_initial_chat_question(query)):
            try:
             answer = gemini_agent(decoded_url,f"{query} && Please answer in a short and concise manner as list and list items (Markdown format)")
             try:
              json_resp = jsonable_encoder(answer)
              return {"answer": json_resp}
             except Exception as e:
              answer = await fallback_chat(query)
              return {"answer": jsonable_encoder(answer)}
            except Exception as e:
             answer = await fallback_chat(query)
             return {"answer": jsonable_encoder(answer)}

        
        # calling casual chat agent
        try:
          casual_chat = await casual_query(query)
        except Exception as e:
          return {"answer": "Sorry, I do not understand your question. Please try again with a different question."}
        # print(f"Casual chat: {casual_chat}")
        if(casual_chat.get("casual_talk")):
            return {"answer": casual_chat.get("response")}

        try:
             answer = qwen_agent(decoded_url,query)
        except Exception as e:
             # print(f"Error qwen agent: {e}")
             try:
                # print("Falling back to gemini agent")
                answer = gemini_agent(decoded_url,query)
             except Exception as e:
              # print(f"Error gemini agent: {e}")
              # calling fallback agent
              answer = await fallback_chat(query, decoded_url)
              return {"answer": answer.get("python_code")}
                 
        # print(f"Final answer: {answer}")
        
        try:
         json_answer = jsonable_encoder(answer)
         return {"answer": json_answer}
        except Exception as e:
         # print(f"Error in jsonable_encoder: {e}")
         # calling fallback agent
         answer = await fallback_chat(query, decoded_url)
        # print(f"Fallback chat: {answer}")
        return {"answer": answer.get("python_code")}
               
    except Exception as e:
       raise HTTPException(status_code=400, detail=f"Error: {e}")
   
   
   
   
   
   
@app.post("/api/csv-chart/")
async def csv_chart(request: dict, authorization: str = Header(None)):

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")

    if token != os.getenv("AUTH_TOKEN"): 
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        query = request.get("query")
        query = query.lower()
        
        csv_url = request.get("csv_url")
        decoded_url = unquote(csv_url)
        
        # initial query handler
        if(if_initial_chart_question(query)):
            try:
             answer = gemini_agent(decoded_url,f"{query} && Create Maximum of 4 charts and Please try to provide all charts in a single image")
             if "temp_chart" in answer:
                 return FileResponse(image_file_path, media_type="image/png")
             else:
                 answer = await fallback_chat(query)
                 return {"answer": answer}
            except Exception as e:
             answer = await fallback_chat(query)
             return {"answer": answer}

        # if not initial query, then continue with the rest of the code

        answer = qwen_agent(decoded_url,query)
                   
        if "temp_chart" in answer:
            return FileResponse(image_file_path, media_type="image/png")
        else:
            try:
                # print("Error in generating final groq_chart output, falling back to smolagent gemini")
                answer = gemini_agent(decoded_url,query)
                if "temp_chart" in answer:
                    return FileResponse(image_file_path, media_type="image/png")
                else:
                    # calling fallback agent
                    answer = await fallback_chat(query, decoded_url)
                    # print(f"Fallback chat: {answer}")
                    return {"answer": answer.get("python_code")}
            except Exception as e:
                # calling fallback agent
                answer = await fallback_chat(query, decoded_url)
                # print(f"Fallback chat: {answer}")
                return {"answer": answer.get("python_code")}
    
    except Exception as e:
       # print(f"Error while generating chart: {e}")
       return FileResponse(image_not_found, media_type="image/png")


