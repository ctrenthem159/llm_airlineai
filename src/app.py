import sys, atexit, json, logging, logging.handlers
from typing import Any
from datetime import date, datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response, ResponseInputParam
from openai.types.conversations import Conversation
from amadeus import Client, Location, ResponseError
import streamlit as st

def setup_logging() -> logging.Logger:
    """ Initalize logger in the console and .log file. """
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    _log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not logger.handlers:
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
amadeus = Client()
client = OpenAI()
MODEL: str = "gpt-4.1-nano"
SYSTEMPROMPT: str = "You are a travel agent chatbot. Your primary goal is to help users find flight deals. IMPORTANT RULE: Before you can search for flights using the 'get_flights' tool, you MUST first translate all specified city names (origin and destination) into their 3-letter IATA codes using the 'get_city' tool. If a user provides a city name, use 'get_city' immediately. The 'get_flights' tool requires IATA codes and an optional departure date (YYYY-MM-DD format). If information is missing or cannot be found, ask the user for clarification."
TOOLS = [
    {
        'type': 'function',
        'name': 'get_city',
        'description': 'Search a cities database and return the IATA code for a given city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'search_term': {
                    'type': 'string',
                    'description': 'A city name or other search term.'
                },
            },
            'required': ['search_term'],
        },
    },
    {
        'type': 'function',
        'name': 'get_flights',
        'description': 'Search for the cheapest flight between two cities.',
        'parameters': {
            'type': 'object',
            'properties': {
                'start_city': {
                    'type': 'string',
                    'description': 'The starting city for the trip.'
                },
                'destination_city': {
                    'type': 'string',
                    'description': 'The destination for the trip.'
                },
                'departure_date': {
                    'type': 'string',
                    'description': 'The date of the flight, using ISO format `YYYY-MM-DD`'
                },
            },
            'required': ['start_city', 'destination_city'],
        },
    },
]

def new_chat() -> str:
    """ Create a new conversation to keep track of inputs & responses.

    Returns:
        The unique ID for this session.
    """
    _conversation: Conversation = client.conversations.create(
        items=[{'type': 'message', 'role': 'system', 'content': SYSTEMPROMPT}]
    )
    logger.info(f'Conversation created successfully. {_conversation}')
    return _conversation.id

def end_chat(conversation: str) -> None:
    """ Deletes the current conversation by ID number.

    Args:
        conversation: The unique ID for the conversation session.
    """
    client.conversations.delete(conversation)
    logger.debug(f'Conversation {conversation} deleted successfully.')

def chat(message: str, conv_id: str) -> str:
    """ Handles the core chat logic, including API calls and tool management.

    The function ensures the model has today's date along with the user's request and sends the request to OpenAI.
    Upon receiving a response, the function will call necessary tools and eventually takes the AI's response and
    returns it to the frontend.

    Args:
        message: The user's input message.
        conv_id: The unique ID for the conversation session.

    Returns:
        The model's response as text.
    """
    _tool_called: bool = False
    _current_date = date.today().isoformat()
    _input: ResponseInputParam = [{'role': 'system', 'content': f'The current date is {_current_date}. Ensure the departure date is today or later.'}, {'role': 'user', 'content': message}]
    logger.debug(f'chat input: {_input}')
    _response: Response = client.responses.create(
        conversation = conv_id,
        model = MODEL,
        tools = TOOLS,
        input = _input
        )
    logger.debug(f'chat response from openAI: {_response}')
    for item in _response.output:
        if item.type == 'function_call':
            _tool_called = True
            logger.debug(f' OpenAI Function Call: {item}')
            arguments = json.loads(item.arguments)
            logger.debug(f'Function arguments: {arguments}')
            if item.name == 'get_city':
                city = get_city(arguments.get('search_term'))
                client.conversations.items.create(
                    conv_id,
                    items = [{
                        'type': 'function_call_output',
                        'call_id': item.call_id,
                        'output': json.dumps({'city_code': city})
                    }]
                )
            if item.name == 'get_flights':
                flight = get_flights(**arguments)
                client.conversations.items.create(
                    conv_id,
                    items = [{
                        'type': 'function_call_output',
                        'call_id': item.call_id,
                        'output': json.dumps({'flight': flight}),
                    }]
                )

    if _tool_called:
        logger.debug("Tool was called, generating new response.")

        _response_second: Response = client.responses.create(
            conversation = conv_id,
            model = MODEL,
            tools = TOOLS,
            input = []
        )
        logger.debug(f'second chat response from openAI: {_response_second}')
        _output: str = _response_second.output_text
    else:
        _output: str = _response.output_text

    return _output

def get_city(search_term: str) -> str | None:
    """
    Accepts a search term from the chat session and looks up the correct IATA code for that city.
    Uses Amadeus Airport & Cities Search API to find the most relevant code.

    Args:
        search_term: A string provided by the user or model.

    Returns:
        location: The IATA code pulled from the first result of the search.
    """
    try:
        # Validate search term input
        if not search_term or len(search_term.strip()) < 3:
            logger.warning(f'Search term {search_term} invalid.')
            return None

        logger.debug(f'Airport search requested: {search_term}')
        res = amadeus.reference_data.locations.get(
            subType = Location.ANY,
            keyword = search_term,
            view = 'LIGHT'
        )
        locations = res.data
        if locations:
            logger.info(f'Found {len(locations)} matching {search_term}')
            logger.info(f'Using IATA Code {locations[0]['iataCode']} for {locations[0]['name']}')
            return locations[0]['iataCode']
        else:
            raise ValueError(f'No results found for search {search_term}')
    except ResponseError as e:
        logger.error(f'An error with Amadeus occured: {e}')
    except Exception as e:
        logger.error(f'An unexpected error occured: {e}')

def get_flights(start_city: str, destination_city: str, departure_date: str = date.today().isoformat()):
    """
    Finds the cheapest flight between two cities and submits the key data on that flight back to the model.

    Args:
        start_city: An IATA code representing the takeoff location (city or airport).
        destination_city: An IATA code representing the destination.
        departure_date: An optional input to select a specific date for the flight. Defaults to today's date if not provided.
            Date must be presented in ISO format (YYYY-MM-DD).

    Returns:
        flight: A dictionary response containing the airline, flight number, aircraft model, flight duration, and lowest ticket price.
    """
    try:
        today = date.today()
        try:
            input_date = date.fromisoformat(departure_date)
        except ValueError:
            logger.error(f"Invalid date format received: {departure_date}. Using today's date.")
            input_date = today

        # Check if the date is in the past. If so, override it.
        if input_date < today:
            logger.warning(f"Model provided past date: {departure_date}. Overriding with current date: {today.isoformat()}")
            validated_date_iso = today.isoformat()
        else:
            validated_date_iso = input_date.isoformat()

        res = amadeus.shopping.flight_offers_search.get(
            originLocationCode = start_city,
            destinationLocationCode = destination_city,
            departureDate = validated_date_iso,
            adults = 1,
            currencyCode = 'USD'
        )
        logger.debug(f'Flight API response: {res}')

        cheapest_flight = None
        min_price = float('inf')
        flight_data = res.data

        for offer in flight_data:
            current_price = float(offer['price']['total'])
            if current_price < min_price:
                min_price = current_price
                cheapest_flight = offer

        if cheapest_flight:
            first_segment = cheapest_flight['itineraries'][0]['segments'][0]
            total_duration = cheapest_flight['itineraries'][0]['duration']

            return {
                'carrier_code': first_segment.get('carrierCode'),
                'flight_number': first_segment.get('number'),
                'airplane_id': first_segment.get('aircraft', {}).get('code'),
                'price_total': cheapest_flight['price']['total'],
                'price_currency': cheapest_flight['price']['currency'],
                'duration': total_duration
            }

    except ResponseError as e:
        logger.error(f'An error with Amadeus occured: {e}')
    except Exception as e:
        logger.error(f'An unexpected error occured: {e}')

def cleanup() -> None:
    """
    Helper function to end the conversation session and ensure everything is properly closed out.
    """
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