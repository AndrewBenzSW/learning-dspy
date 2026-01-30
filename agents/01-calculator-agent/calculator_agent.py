"""
Calculator Agent - A simple DSPy agent that can perform arithmetic operations
"""

import os
import dspy

# AWS credentials are configured via environment variables (AWS_PROFILE, AWS_REGION_NAME)
# Set these in your shell or .env file before running

# Configure which LLM to use
# bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0
# bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
lm = dspy.LM(
  "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
  max_tokens=4000
)
dspy.configure(lm=lm)

# Define the tools our agent can use
def add(a: float, b: float) -> float:
  """Add two numbers together."""
  return a + b

def subtract(a: float, b: float) -> float:
  """Subtract b from a."""
  return a - b

def multiply(a: float, b: float) -> float:
  """Multiply two numbers together."""
  return a * b

def divide(a: float, b: float) -> float:
  """Divide a by b."""
  if b == 0:
    return float('inf')
  return a / b

# Create the agent using DSPy's ReAct module
# ReAct = "Reasoning + Acting" - the agent thinks, acts, observes, repeats
calculator_agent = dspy.ReAct(
  "history, question -> answer: float",
  tools=[add, subtract, multiply, divide],
)

if __name__ == "__main__":
  print("Calculator Agent")
  print("Type 'quit' to exit")
  print("-" * 40)

  history = []

  while True:
    question = input("\nYour question: ")
    if question.lower() in ('quit', 'exit', 'q'):
      print("Goodbye!")
      break

    if question.lower() == 'clear':
      history = []
      print("Memory cleared.")
      continue

    # Format history as a string for the agent
    history_str = "\n".join(history) if history else "No previous calculations."

    result = calculator_agent(history=history_str, question=question)
    print(f"Answer: {result.answer}")

    # Debug: show what the LLM did
    # lm.inspect_history(n=1)

    # Add this exchange to history
    history.append(f"Q: {question} -> A: {result.answer}")
