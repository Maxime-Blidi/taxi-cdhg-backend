from fastapi import APIRouter, HTTPException, Query, status, Response
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from uuid import uuid4
from datetime import datetime, timedelta, timezone
import base64
from typing import Optional
import boto3
import re
import requests
import jwt
import json
from pydantic import BaseModel

from database import DatabaseRelationalPostgreSQL 
from ..models import  table
from constants import AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET, AWS_REGION_NAME, \
                    POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_SCHEMA, POSTGRES_USER

router = APIRouter()


db = DatabaseRelationalPostgreSQL(schema=POSTGRES_SCHEMA,
                                    host=POSTGRES_HOST,
                                    user=POSTGRES_USER,
                                    password=POSTGRES_PASSWORD,
                                    connection_timeout=10,
                                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                                    aws_secret_access_key=AWS_ACCESS_KEY_SECRET,
                                    aws_region_name=AWS_REGION_NAME)

class Response(BaseModel):
    id_trajet: str
    child_id: str
    demand_type: str
    
    start_location: str
    start_time: str
    arrival_location: str
    arrival_time: str
    
    distance: str
    

@router.get("/planning")
async def register(user=Query(),
                   day=Query()):

    return None