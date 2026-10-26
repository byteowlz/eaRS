"""Shadow-only CLM-8B trial on the frozen synthetic challenge.

Uses the same question, parser and summarizer as the Rust sys1/Laya trial;
only the model identity and local endpoint differ. Never connects to eaRS.
"""

import compare_sys1 as evaluator


def main():
    evaluator.MODEL = "clm-latest"
    evaluator.main(
        default_url="http://127.0.0.1:18700/v1/systemone",
        default_output=evaluator.HERE / "artifacts/compare-clm.jsonl",
    )


if __name__ == "__main__":
    main()
