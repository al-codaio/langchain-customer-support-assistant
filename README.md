# LangChain Customer Support Assistant Template

<img src="https://github.com/user-attachments/assets/a3c0153a-95d1-406b-99ca-75c818261750" height="600" />


This is a super simple template for a customer support assistant chatbot. The goal is to show an end-to-end AI application using the various platforms in the LangChain ecosystem including LangChain, LangGraph, LangSmith, and LangGraph Platform. The "knowledge" is all in [knowledge_base.json](knowledge_base.json). There is no vector store so there's no semantic search. The goal is to get a prototype working in your browser. If you just want to test locally (terminal), you won't need [index.html](index.html), [langgraph.json](langgraph.json), and the [server.py](server.py) files.

## 🚀 Template Features

- **Routing**: Directs complex queries to human agents or specialized AI modules.
- **Knowledge Base Integration**: Fetches information from a pre-defined knowledge base ([knowledge_base.json](knowledge_base.json) in this case).
- **Conversational Memory**: Maintains context throughout the interaction.
- **Human Handoff**: Transitions to a human agent when needed.

## ⚡ How To Use
**1. Clone the repo**
```bash
git clone https://github.com/al-codaio/langchain-customer-support-assistant.git
cd langchain-customer-support-assistant
```

**2. Setup and run virtual Python environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install requirements**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI and LangSmith API keys**
Get your OpenAI API key [here](https://platform.openai.com/api-keys) and your LangSmith API key (follow instructions [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key)). Before running these commands in terminal, add the keys to the `OPEN_API_KEY` and `LANGCHAIN_API_KEY` variables below:
```bash
touch .env
echo 'OPENAI_API_KEY="PASTE-YOUR-OPENAI-API-KEY-HERE"' > .env
echo 'LANGCHAIN_API_KEY="PASTE-YOUR-LANGCHAIN-API-KEY-HERE"' >> .env
echo 'LANGCHAIN_TRACING_V2="true"' >> .env
echo 'LANGCHAIN_PROJECT="Customer Support Assistant Template"' >> .env
```

**5. Run the assistant locally in terminal**
```bash
python main.py
```
You should see "Customer Support Assistant: Hello! How can I help you today?" in your terminal. Try asking a question like "What is your refund policy?" and you should see tool calls and the response:

<img width="900" alt="agent-response" src="https://github.com/user-attachments/assets/e9047551-f8d2-4715-b4a8-3809bb93d37a" />

That's pretty much it! You have a customer support assistant using LangChain and LangGraph. The next steps are optional but make the application more interactive since you're using it in your browser versus terminal.

## Optional: Interact with the assistant in Chrome
**6. Interact with the assistant in [LangServe playground](https://python.langchain.com/docs/langserve/#playground)**

Comment out these last two lines of code in [main.py](main.py):
```python
# Comment out the lines below if you will be deploying to LangGraph Platform or running on a local server
if __name__ == "__main__":
    run_assistant()
```
Back in terminal, run the command:
```bash
python server.py
```
Then go to [http://localhost:8000/customer-support/playground/](http://localhost:8000/customer-support/playground/).
You should see the below. I couldn't figure out why "Human Handoff Requested" and "Retry Count" are required, but just check and uncheck "Human Handoff Requested" and give some random number to the "Retry Count." Type in the original query in the messages field "What is your refund policy?" and you can see the full output and all the steps the application took:

<img width="500" src="https://github.com/user-attachments/assets/0a754f7d-b199-4728-ad47-ca6bd2e4e3ca" />
<img width="500" src="https://github.com/user-attachments/assets/29713e7c-a75e-405f-ad5e-5b92495ac1d9" />

**7. Interact with the assistant in a regular webpage**

The [index.html](index.html) page renders a simple chatbot like the image shown at the top of this README. To initialize it, open up another terminal window and run this command:
```bash
python -m http.server 8001
```
Then go to [http://localhost:8001/index.html](http://localhost:8001/index.html) and you'll be able to chat with the assistant in a more familiar UI.

**8. Deploy with [LangGraph Platform](https://www.langchain.com/langgraph-platform)**

TBD
<img width="1081" alt="LGP-deployment-settings" src="https://github.com/user-attachments/assets/ff3dacd7-7308-4bdc-a3e0-4ce26cdbed6d" />

Deploying a simple customer assistant in LangGraph Platform is a bit overkill, but there are definitely some advantages once you have a more complex application with multiple agents and subgraphs. Some benefits

- **REST API Endpoints**: Exposes your graph as a runnable API.
- **Observability**: Integrates with LangSmith for detailed tracing, monitoring, and debugging (although by running locally with your LangSmith API key you also get tracing)
- **Scalability**: Handles concurrent requests and scales your application.
- **Human-in-the-Loop**: Facilitates processes where human intervention is required.
- **A/B Testing & Evaluation**: Tools to test and improve your agent's performance.
