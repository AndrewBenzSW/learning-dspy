# Learning DSPy Agents

A hands-on tutorial for building AI agents with [DSPy](https://dspy.ai/). Each agent introduces a new pattern, building in complexity.

## Agents

| # | Agent | What You'll Learn |
|---|-------|-------------------|
| 01 | [Calculator Agent](agents/01-calculator-agent/) | **ReAct pattern** - Tools, reasoning loops, conversation memory |
| 02 | [Research Agent](agents/02-research-agent/) | **Decomposition** - Breaking complex questions into sub-tasks |
| 03 | [Function Generator](agents/03-function-generator/) | **Generate-Execute-Iterate** - Self-correcting code generation |
| 04 | [Writing Assistant](agents/04-writing-assistant/) | **Signatures** - Text transformation with ChainOfThought |
| 05 | [TDD Orchestrator](agents/05-tdd-orchestrator/) | **Multi-phase workflows** - RED/GREEN/REFACTOR cycle with isolated LLM calls |

## Getting Started

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install dspy boto3

# Run an agent
python agents/01-calculator-agent/calculator_agent.py
```

## Prerequisites

- Python 3.9+
- AWS credentials configured for Bedrock access:
  ```bash
  export AWS_PROFILE=your-profile-name
  export AWS_REGION_NAME=us-east-1
  ```
