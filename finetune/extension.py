"""Extension points for fine-tuning LLM-backed agent graphs (LoRA/QLoRA, DPO, tool-calling reward models)."""


def describe_llm_finetune_playbook() -> dict:
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
    import json

    print(json.dumps(describe_llm_finetune_playbook(), indent=2))


if __name__ == "__main__":
    main()
