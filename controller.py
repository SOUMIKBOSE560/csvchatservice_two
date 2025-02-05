
from typing import Dict
from urllib.parse import unquote
from fastapi import FastAPI, HTTPException, Header
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from agent_service import fallback_chat
from csv_service import custom_jsonable_encoder
from initial_question_handler import if_initial_chart_question, if_initial_chat_question
from smolagent_hf_gemini import csv_smolagent_hf_gemini as gemini_agent
from smolagent_hf_default_qwen import csv_smolagent_hf_qwen as qwen_agent




# uvicorn controller:app --host localhost --port 8085 --reload
app = FastAPI()
image_file_path = os.getenv("IMAGE_FILE_PATH")
image_not_found = os.getenv("IMAGE_NOT_FOUND")
allowed_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)


@app.get("/ping")
async def root():
    return {"message": "Pong !!"}


@app.post("/api/get-chart")
async def get_image(request: dict, authorization: str = Header(None)):
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
        return FileResponse(image_file_path, media_type="image/png")
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "error"}
        

   
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
             answer = gemini_agent(decoded_url,f"{query} && Create Maximum of 2 charts and Please try to provide these 2 charts in a single image, save the image in {image_file_path}")
             if "temp_chart" in answer:
                 return FileResponse(image_file_path, media_type="image/png")
             else:
                fallbackResp = await fallback_chat(query, csv_url)
                return {"answer": fallbackResp, "responseType": "fallback"}

        # if not initial query, then continue with the rest of the code

        answer = qwen_agent(decoded_url,query)
                   
        if "temp_chart" in answer:
            return FileResponse(image_file_path, media_type="image/png")
        else:
            answer = gemini_agent(decoded_url,query)
            if "temp_chart" in answer:
                return FileResponse(image_file_path, media_type="image/png")
            else:
                fallbackResp = await fallback_chat(query, csv_url)
                return {"answer": fallbackResp, "responseType": "fallback"}
    
    except Exception as e:
       return {"answer":"error"}




@app.post("/api/csv-chat/")
async def csv_chat(request: Dict, authorization: str = Header(None)):
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
        
        # Initial query handler
        if if_initial_chat_question(query):
            try:
                answer = gemini_agent(decoded_url, f"{query} && Please answer in a short and descriptive way.")
                try:
                    json_resp = custom_jsonable_encoder(answer)  # Use custom_jsonable_encoder
                    return {"answer": json_resp}
                except Exception as e:
                    fallbackResp = await fallback_chat(query, csv_url)
                    return {"answer": fallbackResp, "responseType": "fallback"}  # Use custom_jsonable_encoder
            except Exception as e:
                fallbackResp = await fallback_chat(query, csv_url)
                return {"answer": fallbackResp, "responseType": "fallback"}  # Use custom_jsonable_encoder
         
        try:
            answer = qwen_agent(decoded_url, query)
        except Exception as e:
            print(f"Error qwen agent: {e}")
            try:
                answer = gemini_agent(decoded_url, query)
            except Exception as e:
                print(f"Error gemini agent: {e}")
                fallbackResp = await fallback_chat(query, csv_url)
                return {"answer": fallbackResp, "responseType": "fallback"}  # Use custom_jsonable_encoder
                 
        try:
            json_answer = custom_jsonable_encoder(answer)  # Use custom_jsonable_encoder
            return {"answer": json_answer}
        except Exception as e:
            print(f"Error in custom_jsonable_encoder: {e}")
            fallbackResp = await fallback_chat(query, csv_url)
            return {"answer": fallbackResp, "responseType": "fallback"}  # Use custom_jsonable_encoder
               
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "error"}
    
    
    
    
# @app.post("/api/csv-fallback-chat/")
# async def csv_chat(request: Dict, authorization: str = Header(None)):
#     if not authorization:
#         raise HTTPException(status_code=401, detail="Authorization header missing")

#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid authorization header format")

#     token = authorization.split(" ")[1]
#     if not token:
#         raise HTTPException(status_code=401, detail="Token missing")

#     if token != os.getenv("AUTH_TOKEN"):  
#         raise HTTPException(status_code=403, detail="Invalid token")

#     try:
#         query = request.get("query")
#         query = query.lower()
        
#         csv_url = request.get("csv_url")
#         decoded_url = unquote(csv_url)
        
#         # Fallback to fallback_chat in case of any error
#         fallbackResp = await fallback_chat(query, csv_url)
#         return {"answer": fallbackResp, "responseType": "fallback"}
               
#     except Exception as e:
#         print(f"Error: {e}")
#         return {"answer": "error"}