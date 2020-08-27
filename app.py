import numpy as np
import uvicorn
from bert_serving.client import BertClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

import predict


class UserRequest(BaseModel):
    content: str
    process_type: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.post("/api/get_result/")
async def query_result(user_request: UserRequest):
    if user_request.process_type == "sentiment":
        request_content = [user_request.content, '']
        return {"result":predict.get_prediction(request_content, estimator)[0][2]}
        pass
    elif user_request.process_type == "word_vec":
        bc = BertClient(ip='localhost', port=23333)
        return {"result": np.array2string(bc.encode([user_request.content]))}
        pass
    else:
        raise HTTPException(status_code=400, detail="Unsupported processing method")


if __name__ == "__main__":
    estimator = predict.create_estimator()
    uvicorn.run(app)
