
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from uuid import UUID



class Drivers(BaseModel):
    pass


class Journeys(BaseModel):
    pass