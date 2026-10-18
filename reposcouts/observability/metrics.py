


def build_run_summary(
    *, 
    execution_status: str, 
    retry_count: int, 
    max_retries: int, 
    critic_passed: bool | None,
) -> dict[str, str | int | bool | None]: 
    """
    Returns a summary of this run, which may retry called by critic.
    """

    if execution_status not in {"completed", "cancelled", "failed"}:
        raise ValueError("execution_status must be a terminal status")

    return {
        "execution_status": execution_status,
        "retry_count": retry_count,
        "critic_passed": critic_passed,
        "outcome": (
            "failed" if execution_status == "failed" else
            "cancelled" if execution_status == "cancelled" else
            "unreviewed" if execution_status == "completed" and critic_passed is None else
            "passed" if execution_status == "completed" and critic_passed else
            "retry_exhausted" if execution_status == "completed" and not critic_passed and retry_count >= max_retries else
            "critic_rejected" if execution_status == "completed" and not critic_passed and retry_count < max_retries else ""
        ), 
    }
