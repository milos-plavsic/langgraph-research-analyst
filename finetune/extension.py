"""Extension points for fine-tuning LLM-backed agent graphs (LoRA/QLoRA, DPO, tool-calling reward models)."""

from ml_core import configure_logging

logger = configure_logging(__name__)


def llm_training_guide() -> dict:
    """Return notes for supervised fine-tuning and adapter training."""
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
        "hook_in_this_repo": "Attach PEFT adapters to planner/draft nodes; keep graph state schema stable.",
    }


def main() -> None:
    """Main."""
    import json

    logger.info(json.dumps(llm_training_guide(), indent=2))


if __name__ == "__main__":
    main()
