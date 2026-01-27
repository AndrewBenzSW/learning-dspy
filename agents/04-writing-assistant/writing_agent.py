import dspy
import os

# Configure AWS Bedrock (same as other agents in this repo)
os.environ["AWS_PROFILE"] = "your-aws-profile-name"
os.environ["AWS_REGION_NAME"] = "us-east-1"

# Initialize the LLM
lm = dspy.LM("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", max_tokens=4000)
dspy.configure(lm=lm)


class WritingAssistant(dspy.Signature):
    """Transform text according to the given task instruction."""

    text: str = dspy.InputField(desc="The original text to transform")
    task: str = dspy.InputField(desc="What to do with the text (e.g., 'summarize', 'improve', 'explain', 'make more formal')")
    result: str = dspy.OutputField(desc="The transformed text")


def transform_text(text: str, task: str) -> str:
    """
    Transform text using the writing assistant.

    This wraps the DSPy signature in a simple function that's easy to call
    from our web app later.
    """
    # ChainOfThought makes the LLM reason step-by-step before answering
    # This generally produces better results than a direct call
    assistant = dspy.ChainOfThought(WritingAssistant)
    response = assistant(text=text, task=task)
    return response.result


# Test the agent when run directly
if __name__ == "__main__":
    test_text = """
    The meeting was very long and we talked about many things including
    the budget which is too high and also the timeline which is too short
    and people are worried about both of these things and we need to figure
    out what to do about it soon.
    """

    print("Original text:")
    print(test_text)
    print("\n" + "="*50 + "\n")

    print("Task: summarize")
    result = transform_text(test_text, "summarize in one sentence")
    print(result)

    print("\n" + "="*50 + "\n")

    print("Task: improve")
    result = transform_text(test_text, "rewrite to be clearer and more professional")
    print(result)
