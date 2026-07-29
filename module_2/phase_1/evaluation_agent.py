from dotenv import load_dotenv
import os

from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "tests", ".env"))

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

eval_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(
    openai_api_key,
    eval_persona,
    evaluation_criteria,
    knowledge_agent,
    max_interactions=10,
)

result = evaluation_agent.evaluate(prompt)
print(result)
