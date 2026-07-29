from dotenv import load_dotenv
import os

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "tests", ".env"))

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)
response = agent.respond(prompt)

print(response)
print(
    "Confirmation: The response should follow the provided knowledge (London) "
    "rather than the LLM's inherent knowledge (Paris)."
)
