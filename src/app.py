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

import sys, atexit, logging, logging.handlers
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response, ResponseInputParam
from openai.types.conversations import Conversation
import streamlit as st

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

    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
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
    logger.info(f'Conversation created successfully. {_conversation}')
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

def cleanup() -> None:
    if 'conversation_id' in st.session_state and st.session_state.conversation_id:
        try:
            end_chat(st.session_state.conversation_id)
            logger.info(f'Cleaned up conversation: {st.session_state.conversation_id}')
        except Exception as e:
            logger.error(f'Error during cleanup: {e}')

atexit.register(cleanup)

st.title('Travel Agent Chat')
st.caption('Find the best flight deals with no effort!')

if 'conversation_id' not in st.session_state:
    try:
        st.session_state.conversation_id = new_chat()
        st.session_state.messages = []
        logger.info(f'New conversation started in Streamlit: {st.session_state.conversation_id}')
        st.success(f'Conversation started! ID: {st.session_state.conversation_id[:8]}...')
    except Exception as e:
        st.error(f'Failed to start conversation: {e}')
        logger.error(f'Failed to start conversation: {e}')
        st.stop()

with st.sidebar:
    st.subheader('Session Info')
    st.text(f'Conversation ID:\n{st.session_state.conversation_id[:16]}...')
    st.text(f'Messages: {len(st.session_state.messages)}')

    if st.button('Reset Conversation', type='secondary'):
        try:
            end_chat(st.session_state.conversation_id)
            st.session_state.conversation_id = new_chat()
            st.session_state.messages = []
            logger.info('Conversation reset')
            st.rerun()
        except Exception as e:
            st.error(f'Failed to reset conversation: {e}')

    if st.button('End Chat', type='primary'):
        try:
            end_chat(st.session_state.conversation_id)
            st.session_state.clear()
            st.success('Chat ended successfully!')
            logger.info('Chat ended by user.')
        except Exception as e:
            st.error(f'Unable to end chat: {e}')

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input('Ask about flights...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.write(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Generating response...'):
            try:
                response: str = chat(prompt, st.session_state.conversation_id)
                st.write(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
            except Exception as e:
                st.error(f'Error generating response: {e}')
                logger.error(f'Chat failed to generate response: {e}')

with st.expander('Debug Info'):
    st.json({
        'conversation_id': st.session_state.conversation_id,
        'message_count': len(st.session_state.messages),
        'server_start': str(SERVER_STARTTIME)
    })