"""
Main entry point for running the full multi-agent recommendation system.

This script initializes the complete workflow and runs it for a sample user query.
"""
import sys
from pathlib import Path
import logging

# Add project root to the Python path to allow for absolute imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the compiled workflow app and the state definition
from agent.workflow_graph import app
from agent.state import AgentState

# Configure logging to show the execution flow
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_recommendation_system(user_input: str, phone_number: str):
    """
    Runs the full recommendation system for a given user input and index.

    Args:
        user_input: The user's query string.
        user_index: The user's ID to look up in the database.
    """
    print("\n" + "="*80)
    print(f"🚀  Starting Recommendation for User Phone Number: {phone_number}")
    print(f"📝  User Input: \"{user_input}\"")
    print("="*80 + "\n")

    # Prepare the initial state for the workflow graph
    initial_state: AgentState = {
        "phone_number": phone_number,
        "input": user_input,
        "chat_history": []  # Start with an empty history for a single run
    }

    try:
        # Invoke the workflow graph with the initial state.
        # This will run the entire chain of agents.
        final_state = app.invoke(initial_state)

        # Get the final response from the resulting state
        final_response = final_state.get("final_response", "Sorry, an error occurred and I could not generate a response.")

        print("\n" + "="*80)
        print("✅  Final Recommendation Response:")
        print("="*80 + "\n")
        print(final_response)
        print("\n" + "="*80)
        print("🎉  Workflow Completed Successfully!")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"An error occurred during the workflow execution: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)
        print("❌  Workflow Failed!")
        print("="*80 + "\n")


if __name__ == "__main__":
    # --- Define the user query here ---
    # Scenario 1: User with explicit needs
    sample_user_input = "我是老年人，每月预算50元，想办5G和宽带，我经常看视频，流量用的不多，通话时长也要求不高"
    # sample_user_input = "你能为我介绍下智慧家庭套餐吗？"
    sample_user_phone_number = "17354132409"

    # Scenario 2: User with database history (no explicit input)
    # sample_user_input = "给我推荐个套餐吧"
    # sample_user_id = 1

    # Scenario 3: User with mixed needs (explicit input + database history)
    # sample_user_input = "我想办个5G套餐，预算150元左右，流量要多一点"
    # sample_user_id = 5
    
    # Run the system with the defined query and user ID
    run_recommendation_system(sample_user_input, sample_user_phone_number)

