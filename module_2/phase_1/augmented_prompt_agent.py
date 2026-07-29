from dotenv import load_dotenv
import os

from workflow_agents.base_agents import AugmentedPromptAgent

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "tests", ".env"))

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)

print(augmented_agent_response)

# The agent uses general knowledge from the LLM (gpt-3.5-turbo) to answer factual questions.
# The college professor persona affects tone and style (e.g., starting with "Dear students,")
# while the underlying factual content still comes from the model's training data.
