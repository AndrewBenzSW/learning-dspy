"""
Function Generator Agent - Generates Python functions from descriptions,
executes them to verify they work, and iterates if there are errors.

This demonstrates the generate → execute → iterate pattern.
"""

import os
import dspy
import sys
from io import StringIO

# AWS credentials are configured via environment variables (AWS_PROFILE, AWS_REGION_NAME)
# Set these in your shell or .env file before running

# Use Haiku for speed during iteration
lm = dspy.LM(
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens=4000
)
dspy.configure(lm=lm)


def execute_python(code: str, test_code: str = "") -> str:
  """
  Execute Python code and return the output or error

  Args:
    code: The function code to execute
    test_code: Optional test code to run after defining the function

  Returns:
    A string with the execution result or error message
  """
  # Combine function definition with test code
  full_code = code
  if test_code:
    full_code = f"{code}\n\n{test_code}"

  # Capture stdout
  old_stdout = sys.stdout
  sys.stdout = StringIO()

  try:
    # Execute in isolated namespace
    namespace = {}
    exec(full_code, namespace)

    output = sys.stdout.getvalue()
    return f"SUCCESS:\n{output}" if output else "SUCCESS: Code executed without errors."

  except Exception as e:
    return f"ERROR: {type(e).__name__}: {e}"

  finally:
    sys.stdout = old_stdout


class GenerateFunction(dspy.Signature):
  """Generate a Python function based on a description."""

  description: str = dspy.InputField(desc="What the function should do")
  function_name: str = dspy.InputField(desc="Name for the function")
  code: str = dspy.OutputField(desc="Complete Python function code (no markdown, just code)")


class FixCode(dspy.Signature):
  """Fix Python code that has an error."""

  code: str = dspy.InputField(desc="The broken code")
  error: str = dspy.InputField(desc="The error message")
  fixed_code: str = dspy.OutputField(desc="The corrected Python code (no markdown, just code)")


class GenerateTestCode(dspy.Signature):
  """Generate test code that calls an already-defined function."""

  description: str = dspy.InputField(desc="What the function does")
  function_name: str = dspy.InputField(desc="Name of the function to test")
  test_code: str = dspy.OutputField(desc="Python code that ONLY calls the function and prints results.  "
    "Do NOT redefine the function - it already exists (no markdown, just code)")


class FunctionGenerator(dspy.Module):
  """Generate Python functions and verifies they work."""

  def __init__(self, max_attempts: int = 3):
    super().__init__()
    self.generate = dspy.ChainOfThought(GenerateFunction)
    self.generate_test = dspy.ChainOfThought(GenerateTestCode)
    self.fix = dspy.ChainOfThought(FixCode)
    self.max_attempts = max_attempts

  def forward(self, description: str, function_name: str) -> dict:
    # Step 1
    print(f"\nGenerating function '{function_name}'...")
    result = self.generate(
      description=description,
      function_name=function_name
    )
    code = result.code

    # Step 2: Generate test code
    print("Generating test code...")
    test_result = self.generate_test(
      description=description,
      function_name=function_name
    )
    test_code = test_result.test_code

    # Step 3: Execute and iterate
    for attempt in range(1, self.max_attempts + 1):
      print(f"\nAttempt {attempt}/{self.max_attempts}: Executing...")

      output = execute_python(code, test_code)

      if output.startswith("SUCCESS"):
        print(output)
        return {
          "success": True,
          "code": code,
          "test_code": test_code,
          "output": output,
          "attempts": attempt
        }

      # Execution failed - try to fix
      print(output)

      if attempt < self.max_attempts:
        print("Attempting to fix...")
        fix_result = self.fix(code=code, error=output)
        code = fix_result.fixed_code

    # All attempts failed
    return {
      "success": False,
      "code": code,
      "test_code": test_code,
      "otuput": output,
      "attempts": self.max_attempts
    }


if __name__ == "__main__":
  print("Function Generator Agent")
  print("Describe a function and I'll generate, test, and fix it.")
  print("Type 'quit' to exit")
  print("-" * 60)

  agent = FunctionGenerator(max_attempts=3)

  while True:
    print()
    description = input("What should the function do? ")

    if description.lower() in ('quit', 'exit', 'q'):
      print("Goodbye!")
      break

    function_name = input("Function name: ")

    result = agent(description=description, function_name=function_name)

    print(f"\n{'='*60}")
    if result["success"]:
      print(f"SUCCESS after {result['attempts']} attempt(s)!")
    else:
      print(f"FAILED after {result['attempts']} attempts")

    print(f"\nGENERATED CODE:\n")
    print(result["code"])

    print(f"\nTEST CODE:\n")
    print(result["test_code"])
    print('=' * 60)
