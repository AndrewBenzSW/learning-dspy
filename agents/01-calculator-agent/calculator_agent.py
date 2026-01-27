"""
Calculator Agent - A simple DSPy agent that can perform arithmetic operations
"""

import os
import dspy

# Configure AWS credentials via environment variables
os.environ["AWS_PROFILE"] = "your-aws-profile-name"
os.environ["AWS_REGION_NAME"] = "us-east-1"

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
  "question -> answer: float",
  tools=[add, subtract, multiply, divide],
)

if __name__ == "__main__":
  print("Calculator Agent")
  print("Type 'quit' to exit")
  print("-" * 40)

  while True:
    question = input("\nYour question: ")
    if question.lower() in ('quit', 'exit', 'q'):
      print("Goodbye!")
      break

    result = calculator_agent(question=question)
    print(f"Answer: {result.answer}")
