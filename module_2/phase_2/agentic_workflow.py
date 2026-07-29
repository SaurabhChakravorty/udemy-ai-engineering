# agentic_workflow.py

# TODO: 1 - Import agents from workflow_agents.base_agents
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

import os
from dotenv import load_dotenv

# TODO: 2 - Load OpenAI API key into openai_api_key
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "tests", ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")

# TODO: 3 - Load Product-Spec-Email-Router.txt into product_spec
with open(os.path.join(os.path.dirname(__file__), "workflow_agents", "Product-Spec-Email-Router.txt"), "r") as f:
    product_spec = f.read()

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

# TODO: 4 - Instantiate action_planning_agent
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    + product_spec
)

# TODO: 6 - Instantiate product_manager_knowledge_agent
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_product_manager,
    knowledge_product_manager,
)

# Product Manager - Evaluation Agent
# TODO: 7 - Instantiate product_manager_evaluation_agent
persona_pm_eval = "You are an evaluation agent that checks the answers of other worker agents"
eval_criteria_pm = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]. "
    "Each story must be specific to the Email Router product (email ingestion, classification, "
    "routing, RAG responses, dashboards, knowledge base, SMEs, customer support, IT admin, "
    "security/compliance). Reject generic or unrelated product stories."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_pm_eval,
    eval_criteria_pm,
    product_manager_knowledge_agent,
    max_interactions=10,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups. "
    "Only define features for the Email Router product described in the specification below.\n\n"
    + product_spec
)

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_program_manager,
    knowledge_program_manager,
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate program_manager_evaluation_agent
evaluation_criteria_program = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user. "
    "Each feature must be grounded in the Email Router product specification "
    "(email ingestion, classification, RAG, routing to SMEs, dashboards, knowledge base, "
    "security/RBAC/MFA/GDPR). Reject generic features such as Social Sharing, Dark Mode, "
    "or In-App Messaging unless explicitly supported by the Email Router spec."
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program,
    program_manager_knowledge_agent,
    max_interactions=10,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story. "
    "Only create engineering tasks for the Email Router product described in the specification below.\n\n"
    + product_spec
)

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_dev_engineer,
    knowledge_dev_engineer,
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 9 - Instantiate development_engineer_evaluation_agent
evaluation_criteria_dev = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first. "
    "Each task must implement Email Router functionality from the product specification. "
    "Reject placeholder tasks such as 'Implement feature X' or generic unrelated work."
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev,
    development_engineer_knowledge_agent,
    max_interactions=10,
)


# TODO: 11 - Define support functions
def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(query, initial_response=response)
    return result["final_response"]


def program_manager_support_function(query):
    response = program_manager_knowledge_agent.respond(query)
    result = program_manager_evaluation_agent.evaluate(query, initial_response=response)
    return result["final_response"]


def development_engineer_support_function(query):
    response = development_engineer_knowledge_agent.respond(query)
    result = development_engineer_evaluation_agent.evaluate(query, initial_response=response)
    return result["final_response"]


# TODO: 10 - Instantiate routing_agent
routing_agent = RoutingAgent(openai_api_key, [])
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible for defining product personas and user stories only. "
            "Does not define features or tasks. Does not group stories"
        ),
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible for defining product features by grouping related user stories. "
            "Does not create stories or tasks."
        ),
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible for defining detailed engineering tasks with acceptance criteria "
            "and estimations. Does not create stories or features."
        ),
        "func": lambda x: development_engineer_support_function(x),
    },
]

# Run the workflow
print("\n*** Workflow execution started ***\n")

workflow_prompt = (
    "Create a complete Email Router development plan including user stories, "
    "product features, and detailed engineering tasks."
)
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")
print("\nDefining workflow steps for the Email Router project plan")

# TODO: 12 - Implement the workflow with explicit phases and accumulated context
workflow_steps = [
    "Generate Email Router user stories from the product specification.",
    "Define Email Router features from the generated user stories.",
    "Create detailed Email Router engineering tasks for those features.",
]

completed_steps = []
context = f"Product specification:\n{product_spec}\n"

for step in workflow_steps:
    routed_query = (
        f"Use only the Email Router product context below.\n\n"
        f"{context}\n\nCurrent task: {step}\n"
        "Return the required structured artifact only."
    )
    print(f"\nProcessing step: {step}")
    result = routing_agent.route(routed_query)
    if result is None:
        result = "No result returned for this step."
    completed_steps.append(result)
    context += f"\nOutput for {step}:\n{result}\n"
    print(f"Step result:\n{result}")

print("\n*** Workflow completed ***")
print("\nFinal Email Router project plan:")
print("\n\n".join(completed_steps) if completed_steps else "No steps completed.")
