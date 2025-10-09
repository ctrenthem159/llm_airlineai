# TODO hook Amadeus API for flight prices
# TODO hook Aviationstack API for flight records
# TODO build openAI tool to talk to Amadeus
# TODO implement Redis for AI
# TODO implement deployment

# OpenAI
# user input: start & end cities
# tools: amadeus and aviationstack apis
# output: best beal
# considerations: is the user flexible on airline/plane type/dates? expand search if yes

import sys, logging, logging.handlers
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response, ResponseInputParam
from openai.types.conversations import Conversation

def setup_logging() -> logging.Logger:
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    _log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    _log_file_handler = logging.handlers.TimedRotatingFileHandler(
    '.log',
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
    )
    _log_file_handler.setLevel(logging.DEBUG)
    _log_file_handler.setFormatter(_log_formatter)
    logger.addHandler(_log_file_handler)

    _log_console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    _log_console_handler.setLevel(logging.INFO)
    _log_console_handler.setFormatter(_log_formatter)
    logger.addHandler(_log_console_handler)

    return logger

logger: logging.Logger = setup_logging()
SERVER_STARTTIME: datetime = datetime.now(timezone.utc)

load_dotenv()
client = OpenAI()
MODEL: str = "gpt-4.1-nano"
SYSTEMPROMPT: str = "You are a customer service agent for a travel agency. Provide customers with friendly, helpful service to find the best flight deals for their upcoming trip. Using the user's city combination and intended travel date, look for the best possible flight deal. If you cannot find the information, say you do not know and suggest that the user consider different dates or cities."

def new_chat() -> str:
    _conversation: Conversation = client.conversations.create(
        items=[{'type': 'message', 'role': 'system', 'content': SYSTEMPROMPT}]
    )
    logger.debug(f'Conversation created successfully. {_conversation}')
    return _conversation.id

def end_chat(conversation: str) -> None:
    client.conversations.delete(conversation)
    logger.debug(f'Conversation {conversation} deleted successfully.')

def chat(message: str, conv_id: str) -> str:
    _input: ResponseInputParam = [{'role': 'user', 'content': message}]
    logger.debug(f'chat input: {_input}')
    _response: Response = client.responses.create(
        conversation = conv_id,
        model = MODEL,
        input = _input
        )
    logger.debug(f'chat response from openAI: {_response}')
    _output: str = _response.output_text
    return _output