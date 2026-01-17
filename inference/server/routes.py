import asyncio
import jax
import jax.numpy as jnp
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from server.schemas import InferenceCompletion, InferenceCompletionResponse
import traceback
import datetime
from server.dependencies import get_model

router = APIRouter(prefix = "/v1", tags = ["server"])

@router.post("/inference/completions")
async def inference_completion(request: InferenceCompletion):
    
    try: 
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = str(traceback.format_exc())
        )
     
#    if request.stream:
#        return StreamingResponse(
#            stream_completion( # TODO  - implemenet func that streams tokens into the StreamingResponse class
#            )
#        )

   
#    else:
#        response_content = await generate_completion() # TODO implement func that returns an entire generation
#        total_inference_time = response_content["inf_time"] 
#        return InferenceCompletionResponse(
#            response = response_content,
#            date_finished = str(datetime.datetime.now()),
#            total_inference_time = total_inference_time
#        )