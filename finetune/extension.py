"""Extension points for fine-tuning LLM-backed agent graphs (LoRA/QLoRA, DPO, tool-calling reward models)."""

from ml_core import configure_logging

logger = configure_logging(__name__)


def describe_llm_finetune_playbook() -> dict:
    """Execute the describe llm finetune playbook routine."""
    return {
        "stack": ["transformers", "peft", "trl", "datasets"],
        "typical_workflows": [
            "Supervised fine-tune (SFT) on curated research QA pairs.",
            "Preference optimization (DPO/ORPO) using critic vs final report pairs.",
            "Adapter-only training (LoRA) to preserve base model and cut GPU memory.",
        ],
        "evaluation": [
            "Citation precision/recall on held-out sources.",
            "Tool-call success rate in LangGraph replay harness.",
        ],
        "hook_in_this_repo": "Replace stub nodes with `ChatOpenAI` + PEFT model id; keep graph state schema stable.",
    }


def main() -> None:
    """Execute the main routine."""
    import json

    logger.info(json.dumps(describe_llm_finetune_playbook(), indent=2))


if __name__ == "__main__":
    main()
