"""
TDD Orchestrator Agent

An agent that enforces a strict RED -> GREEN -> REFACTOR cycle
by launching isolated LLM calls for each phase
"""

import os
import subprocess
from pathlib import Path

# AWS Bedrock configuration
os.environ["AWS_PROFILE"] = "your-aws-profile-name"
os.environ["AWS_REGION_NAME"] = "us-east-1"

import dspy

# The test project we'll work with
PROJECT_DIR = Path(__file__).parent / "test-project"


# ================================================================
# TOOLS - These are the only actions the LLM can take
# ================================================================

def run_tests() -> dict:
  '''
  Run the test suite.  Returns structured results.
  The LLM doesn't know this runs vitest - it just gets pass/fail
  '''
  result = subprocess.run(
    ["pnpm", "test"],
    cwd=PROJECT_DIR,
    capture_output=True,
    text=True,
    timeout=30
  )

  return {
    "success": result.returncode == 0,
    "output": result.stdout + result.stderr
  }


def read_file(filepath: str) -> dict:
  """
  Read a file from the project.
  Returns the content or an error message.
  """
  full_path = PROJECT_DIR / filepath
  try:
    content = full_path.read_text()
    return {"success": True, "content": content}
  except FileNotFoundError:
    return {"success": False, "error": f"File not found: {filepath}"}
  except Exception as e:
    return {"success": False, "error": str(e)}


def write_file(filepath: str, content: str) -> dict:
  """
  Write content to a file in the project.
  Creates parent directories if needed.
  """
  full_path = PROJECT_DIR / filepath
  try:
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    return {"success": True, "path": filepath}
  except Exception as e:
    return {"success": False, "error": str(e)}


def list_files() -> dict:
  """
  List all source and test files in teh rpoject.
  Excludes node_modules and other noise.
  """
  files = []
  for path in PROJECT_DIR.rglob("*"):
    if path.is_file() and "node_modules" not in str(path):
      relative = path.relative_to(PROJECT_DIR)
      files.append(str(relative))
  return {"files": sorted(files)}


# ================================================================
# DSPy SIGNATURES - Define what each phase does
# ================================================================

class WriteFailingTest(dspy.Signature):
  """
  You are in the RED phase of TDD. Your ONLY job is to write a failing test.

  Rules:
  - Write exactly ONE test that captures the requirement
  - The test MUST fail (the feature doesn't exist yet)
  - Use vitest syntax (describe, it, expectes)
  - Put tests in a .test.js file
  """
  requirement: str = dspy.InputField(desc="What the code should do")
  current_files: str = dspy.InputField(desc="Files currently in the project")
  test_code: str = dspy.OutputField(desc="The test file content")
  test_filepath: str = dspy.OutputField(desc="Where to save the test (e.g., src/math.test.js)")


class WriteMinimalCode(dspy.Signature):
  """
  You are in teh GREEN phase of TDD. Your ONLY job is to make the test pass.

  Rules:
  - Write the MINIMUM code to make the test pass
  - Do NOT add extra features or edge cases
  - Do NOT refactor or clean up
  - Just make it work, nothing more
  """
  test_code: str = dspy.InputField(desc="The failing test")
  test_error: str = dspy.InputField(desc="The error message from running the test")
  implementation_code: str = dspy.OutputField(desc="The code that makes the test pass")
  implementation_filepath: str = dspy.OutputField(desc="Where to save the code (e.g., src/math.js)")


class RefactorCode(dspy.Signature):
  """
  You are in the REFACTOR phase of TDD. Clean up while keeping tests green.

  Rules:
  - Improve code quality (naming, structure, clarity)
  - Do NOT change behavior
  - Do NOT add new features
  - Tests must still pass after your changes
  - If code is already clean, return it unchanged
  """
  test_code: str = dspy.InputField(desc="The test (behavior that must not change)")
  implementation_code: str = dspy.InputField(desc="The current implementation")
  refactored_code: str = dspy.OutputField(desc="The improved implementation")
  changes_made: str = dspy.OutputField(desc="Brief description of what you improved, or 'No changes needed'")

# ================================================================
# PHASE EXECUTORS - Run each phase of TDD
# ================================================================

def execute_red_phase(requirement: str) -> dict:
  '''
  RED phase: Get teh LLM to write a failing test.

  Returns success only if:
  1. LLM produces a test
  2. Test file is written
  3. Tests actually FAIL (That's the goal!)
  '''
  # Get current project state
  files = list_files()
  files_str = "\n".join(files["files"]) if files["files"] else "(empty project)"

  # Ask LLM to write a failing test
  writer = dspy.Predict(WriteFailingTest)
  result = writer(requirement=requirement, current_files=files_str)

  # Write the test file
  write_result = write_file(result.test_filepath, result.test_code)
  if not write_result["success"]:
    return {"success": False, "error": f"Failed to write test: {write_result['error']}"}

  # Run tests - we WANT them to fail
  test_result = run_tests()

  if test_result["success"]:
    return {
      "success": False,
      "error": "Tests passed! But we're in RED phase - tests should FAIL.",
      "output": test_result["output"]
    }

  return {
    "success": True,
    "phase": "RED",
    "message": "Test written and failing as expected",
    "test_file": result.test_filepath,
    "output": test_result["output"]
  }


def execute_green_phase(test_filepath: str, max_retries: int = 3) -> dict:
  """
  GREEN phase: Write minimal code to make the test pass.

  Retries up to max_retries times if the implementation doesn't work,
  feeding the error back to the LLM each time.
  """
  # Read the test file
  test_content = read_file(test_filepath)
  if not test_content["success"]:
    return {"success": False, "error": f"Can't read test: {test_content['error']}"}

  # Track attempts for feedback
  previous_attempts = []

  for attempt in range(1, max_retries + 1):
    print(f"    Attempt {attempt}/{max_retries}...")

    # Get current test output (the error message)
    test_result = run_tests()

    # Build context from previous failures
    hint = ""
    if previous_attempts:
      hint = "\n\nPREVIOUS FAILED ATTEMPTS:\n"
      for i, prev in enumerate(previous_attempts, 1):
        hint += f"\n--- Attempt {i} ---\n"
        hint += f"Code tried:\n{prev['code'][:500]}...\n"
        hint += f"Error: {prev['error'][:300]}\n"
      hint += "\nDo NOT repeat these mistakes. Try a different approach."

    # Ask LLM to write implementation
    writer = dspy.Predict(WriteMinimalCode)
    result = writer(
      test_code=test_content["content"],
      test_error=test_result["output"] + hint
    )

    # Write the implementation
    write_result = write_file(result.implementation_filepath, result.implementation_code)
    if not write_result["success"]:
      return {"success": False, "error": f"Failed to write code: {write_result['error']}"}

    # Run tests - they should PASS now
    test_result = run_tests()

    if test_result["success"]:
      return {
        "success": True,
        "phase": "GREEN",
        "message": f"Implementation written and tests pass (attempt {attempt})",
        "impl_file": result.implementation_filepath,
        "attempts": attempt,
        "output": test_result["output"]
      }

    # Record this failed attempt
    previous_attempts.append({
      "code": result.implementation_code,
      "error": test_result["output"]
    })

  # All retries exhausted
  return {
    "success": False,
    "error": f"Tests still failing after {max_retries} attempts",
    "attempts": max_retries,
    "output": test_result["output"]
  }


def execute_refactor_phase(test_filepath: str, impl_filepath: str) -> dict:
  """
  REFACTOR phase: Clean up the code while keeping tests green.

  Returns success only if tests still pass after refactoring.
  """
  # Read both files
  test_content = read_file(test_filepath)
  impl_content = read_file(impl_filepath)

  if not test_content["success"] or not impl_content["success"]:
    return {"success": False, "error": "Can't read test or implementation file"}

  # Ask LLM to refactor
  refactorer = dspy.Predict(RefactorCode)
  result = refactorer(
    test_code=test_content["content"],
    implementation_code=impl_content["content"]
  )

  # Write refactored code
  write_result = write_file(impl_filepath, result.refactored_code)
  if not write_result["success"]:
    return {"success": False, "error": f"Failed to write: {write_result['error']}"}
  
  # Verify tests still pass
  test_result = run_tests()

  if not test_result["success"]:
    # Rollback! Restore original
    write_file(impl_filepath, impl_content["content"])
    return {
      "success": False,
      "error": "Refactoring broke tests - rolled back",
      "output": test_result["output"]
    }
  
  return {
    "success": True,
    "phase": "REFACTOR",
    "message": result.changes_made,
    "output": test_result["output"]
  }


# ================================================================
# MULTI-CYCLE ORCHESTRATION
# ================================================================

def run_tdd_cycles(requirements: list[str]) -> dict:
  """
  Run multiple TDD cycles, one for each requirement.

  Each cycle builds on the code produced by previous cycles.
  Returns a summary of all cycles.
  """
  results = []

  for i, requirement in enumerate(requirements, 1):
    print(f"\n{'='*60}")
    print(f"CYCLE {i}/{len(requirements)}: {requirement}")
    print('='*60)

    cycle_result = {"requirement": requirement, "cycle": i}

    # RED
    print("\n  [RED] Writing failing test...")
    red_result = execute_red_phase(requirement)
    cycle_result["red"] = red_result

    if not red_result["success"]:
      print(f"  [RED] FAILED: {red_result.get('error')}")
      cycle_result["status"] = "failed_red"
      results.append(cycle_result)
      continue  # Skip to next requirement

    print(f"  [RED] OK - Test written: {red_result['test_file']}")

    # GREEN
    print("\n  [GREEN] Making test pass...")
    green_result = execute_green_phase(red_result["test_file"])
    cycle_result["green"] = green_result

    if not green_result["success"]:
      print(f"  [GREEN] FAILED: {green_result.get('error')}")
      cycle_result["status"] = "failed_green"
      results.append(cycle_result)
      continue

    print(f"  [GREEN] OK - Passed on attempt {green_result.get('attempts', 1)}")

    # REFACTOR
    print("\n  [REFACTOR] Cleaning up...")
    refactor_result = execute_refactor_phase(
      red_result["test_file"],
      green_result["impl_file"]
    )
    cycle_result["refactor"] = refactor_result

    if not refactor_result["success"]:
      print(f"  [REFACTOR] FAILED (rolled back): {refactor_result.get('error')}")
      cycle_result["status"] = "failed_refactor"
    else:
      print(f"  [REFACTOR] OK - {refactor_result.get('message')}")
      cycle_result["status"] = "complete"

    results.append(cycle_result)

  # Summary
  completed = sum(1 for r in results if r["status"] == "complete")
  print(f"\n{'='*60}")
  print(f"SUMMARY: {completed}/{len(requirements)} cycles completed")
  print('='*60)

  return {
    "total": len(requirements),
    "completed": completed,
    "cycles": results
  }


# ================================================================
# MAIN - Run multi-cycle TDD
# ================================================================

if __name__ == "__main__":
    lm = dspy.LM("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", max_tokens=4000)
    dspy.configure(lm=lm)

    print("=== TDD Orchestrator - Multi-Cycle Mode ===")

    # Build out a math module with multiple functions
    requirements = [
        "A function called 'add' that adds two numbers",
        "A function called 'subtract' that subtracts the second number from the first",
        "A function called 'multiply' that multiplies two numbers",
    ]

    result = run_tdd_cycles(requirements)

    print(f"\nFinal result: {result['completed']}/{result['total']} requirements implemented")
