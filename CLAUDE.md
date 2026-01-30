# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **tutorial repository** for learning DSPy agent patterns. Users are here to learn, not just get working code.

## Teaching Approach

When helping users build agents:
- **Guide, don't just generate** - Provide code snippets for users to type themselves
- **Explain each piece** - What it does and why we're doing it
- **Build incrementally** - One step at a time, testing as we go
- **Celebrate debugging** - Errors are learning opportunities; use `lm.inspect_history()` to explore

New agents should go in `agents/NN-agent-name/` following the existing numbering.

## Running Agents

Each agent has its own directory. To run:

```bash
source venv/bin/activate
python agents/01-calculator-agent/calculator_agent.py
python agents/02-research-agent/research_agent.py
python agents/03-function-generator/function_generator.py
```

## AWS Bedrock Configuration

Agents use Claude via AWS Bedrock. Before running, set your AWS credentials:

```bash
export AWS_PROFILE=your-profile-name
export AWS_REGION_NAME=us-east-1
```

Or create a `.env` file (see `.env.example`).

LLM initialization uses LiteLLM format:
```python
lm = dspy.LM("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", max_tokens=4000)
dspy.configure(lm=lm)
```

## Agent Patterns

| Agent | Pattern | Key DSPy Concept |
|-------|---------|------------------|
| 01-calculator-agent | ReAct | `dspy.ReAct` with tools, signature strings, conversation history |
| 02-research-agent | Decomposition | `dspy.Signature` classes, `dspy.ChainOfThought`, multi-module pipelines |
| 03-function-generator | Generate-Execute-Iterate | Code execution with `exec()`, self-correcting loops |
