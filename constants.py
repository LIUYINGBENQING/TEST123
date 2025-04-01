import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import openai
from mops.logger import get_logger

load_dotenv(override=True)

data_dir = Path(__file__).parent.parent / "data"

figure_dir = Path(__file__).parent.parent / "figures"

# gpt
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_API_BASE"),
# )
openai_model = os.getenv("OPENAI_MODEL")

# llama n 
# client = OpenAI(
# api_key = XXX,
# base_url = "https://api.llama-api.com"
# )
# openai_model = "llama3.1-70b"

# aliyun
client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"), 
    base_url=os.getenv("ALIYUN_API_BASE"),
)
# openai_model = "qwen2.5-72b-instruct"
aliyun_model = os.getenv("ALIYUN_MODEL")


# # lmdeploy本地部署
# client = OpenAI(
#     base_url='http://0.0.0.0:23333/v1',

#     # required but ignored
#     # api_key='ollama',
# )
# openai_model = client.models.list().data[0].id



# openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-1106")




logger = get_logger(Path(__file__).parent.parent / "log.log")

