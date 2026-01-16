from pydantic import BaseModel, field_validator
from typing import List
from datetime import datetime, timedelta

class InferenceCompletion(BaseModel):
    # request for model inferecne
    message:List[int] # tokenized request
    date_requested: datetime
    stream:bool = True
    
class InferenceCompletionResponse(BaseModel): # model for inference responses
    response:List[int] # tokenized repsonse
    date_finished: str
    total_inference_time: str
    
    @field_validator("total_inference_time")
    @classmethod
    def check_inference_time(cls, v: timedelta):
        if v.total_seconds() <= 0:
            raise ValueError("inference_time cannot be less than 0")
        return v