# AI Travel Agent

This is an intelligent conversational AI tool demonstrating the ability of an LLM to help users find the cheapest airfare for a given flight. It combines OpenAI's GPT-4.1 model with a free travel API to search thousands of available flights daily.

The project comes with support for session management and a built-in UI, ready for deployment with one command.

## Installation

Setting up the project requires a few steps, and then a single command to run:

1. Clone/fork the repo and ensure you have the necessary environment variables set up either in `.env` or in your deployment environment.

```sh
OPENAI_API_KEY="Your OpenAI API key with access to gpt-4.1-nano (pretty much any API key will work)"
AMADEUS_CLIENT_ID="Your API key from Amadeus"
AMADEUS_CLIENT_SECRET="Your API secret from Amadeus"
```

2. Install the prerequisites.

3. Run the app with `poetry run streamlit run ./src/app.py`

At this point, your console will begin logging and a `.log` file will be generated in the project root folder. Streamlit automatically opens a browser window on your computer to the app and you can begin testing before deployment.

### Requirements

This project was built using Poetry, so the easiest method of getting set up is just to install Poetry and use that for dependency management. As an alternative, you can set up your own environment using:

```sh
python -m venv .venv
source .venv/bin/activate # Windows: .venv/Scripts/activate
pip install openai amadeus streamlit python-dotenv
```

## Deployment

You're on your own for deployment. This app is a demo, but it should be fully functional to run in Vercel, Heroku, or similar platforms.

## Usage

Everything happens through the chat interface. You can ask the model look up airport or city codes, and you can also have it perform it's primary function: look up the cheapest airfare for a given city combination.
