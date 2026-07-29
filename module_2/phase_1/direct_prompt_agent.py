from dotenv import load_dotenv
import os

from workflow_agents.base_agents import DirectPromptAgent

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "tests", ".env"))

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

print(direct_agent_response)
print(
    "Knowledge source: The agent uses general knowledge from the selected LLM model "
    "(gpt-3.5-turbo), not external documents or a custom knowledge base."
)
