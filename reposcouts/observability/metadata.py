from typing import Dict

def build_run_metadata(
    *, 
    run_id: str, 
    environment: str,
    agent_version: str,
    model_name: str,
    prompt_version: str,
    retry_count: int = 0,
) -> Dict[str, str | int]: 
    """
    Collects metadata about each user request run.
    Store the metadata in a dictionary for easy access and logging.
    Args: 
        run_id (str): Unique identifier for the run.
        environment (str): The environment in which the run is executed (e.g., "production", "staging").
        agent_version (str): Version of the agent being used. (e.g., "v1.0.0")
        model_name (str): Name of the model being used for evaluation.
        prompt_version (str): Version of the prompt being used.
        retry_count (int, optional): Number of retries for the run. Defaults to 0.
    """

    return {
        "run_id": run_id,
        "environment": environment,
        "agent_version": agent_version,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "retry_count": retry_count,
    }


if __name__ == "__main__":

    # Example usage
    metadata = build_run_metadata(
        run_id="12345",
        environment="production",
        agent_version="v1.0.0",
        model_name="gpt-6.1-sol",
        prompt_version="v1.2.3",
        retry_count=1,
    )
    print(metadata)