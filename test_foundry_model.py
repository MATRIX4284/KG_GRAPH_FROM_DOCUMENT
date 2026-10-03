import json
import os
import re
import sys

import networkx as nx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pyvis.network import Network

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
client = OpenAI(
    base_url="https://kaustav-foundry-reaource.services.ai.azure.com/openai/v1",
    api_key=token_provider,
)
DEPLOYMENT = "gpt-5.1"
response = client.responses.create(
    model=DEPLOYMENT,
    input="What is the capital of France?",
)

print(f"answer: {response.output[0]}")
