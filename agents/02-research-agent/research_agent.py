"""
Research Agent - Breaks complex questions into sub-questions,
researches each, and synthesizes a final answer.

This demonstrates task decomposition - a key pattern for complex reasoning
"""

import os
import dspy

# AWS credentials are configured via environment variables (AWS_PROFILE, AWS_REGION_NAME)
# Set these in your shell or .env file before running

# Use Haiku for speed (we'll make many LLM calls)
lm = dspy.LM(
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens=4000
)
dspy.configure(lm=lm)

# Module 1: Break a complex question into sub-questions
class Decompose(dspy.Signature):
  """Break a complex quesiton into 3-5 simpler sub-questions."""

  question: str = dspy.InputField(desc="The complex question to research")
  sub_questions: list[str] = dspy.OutputField(desc="List of simpler sub-questions")

# Module 2: Answer a single question
class Answer(dspy.Signature):
  """Answer a questions concisely based on your knowledge."""

  question: str = dspy.InputField(desc="The question to answer")
  answer: str = dspy.OutputField(desc="A concise answer")

# Module 3: Synthesize multiple answers into a final response
class Synthesize(dspy.Signature):
  """Combine multiple Q&A paris into a comprehensive final answer."""

  original_question: str = dspy.InputField(desc="The original complex question")
  research: str = dspy.InputField(desc="The sub-questions and their answers")
  final_answer: str = dspy.OutputField(desc="A comprehensive answer to the original question")


class ResearchAgent(dspy.Module):
  """A researchagent that decomposes, researches, and synthesizes"""

  def __init__(self):
    super().__init__()
    # ChainOfThought adds reasoning before each output
    self.decompose = dspy.ChainOfThought(Decompose)
    self.answer = dspy.ChainOfThought(Answer)
    self.synthesize = dspy.ChainOfThought(Synthesize)

  def forward(self, question: str) -> str:
    # Step 1: Break into sub-questions
    decomp_result = self.decompose(question=question)
    sub_questions = decomp_result.sub_questions

    print(f"\nDecomposed into {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
      print(f"  {i}, {sq}")

    # Step 2: Answer each sub-question
    research_parts = []
    print("\nResearching each sub-question:")
    for sq in sub_questions:
      result = self.answer(question=sq)
      research_parts.append(f"Q: {sq}\nA: {result.answer}")
      print(f"   ✓ {sq[:50]}...")

    research = "\n\n".join(research_parts)

    # Step 3: Synthesize into final answer
    print("\nSynthesizing final answer...")
    final = self.synthesize(
      original_question=question,
      research=research
    )

    return final.final_answer


if __name__ == "__main__":
  print("Research Agent")
  print("Ask complex questions - I'll break them down and research each part.")
  print("Type 'quit' to exit")
  print("-" * 60)

  agent = ResearchAgent()

  while True:
    question = input("\nYour question: ")

    if question.lower() in ('quit', 'exit', 'q'):
      print("Goodbye!")
      break

    answer = agent(question=question)
    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:\n{answer}")
    print('=' * 60)
