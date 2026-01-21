# Router Prompt Template

## Task
You are an assistant that determines how many hops a question requires.
- 1-hop: one direct fact answers the question.
- 2-hop: two linked facts are needed.
- 3-hop: three linked facts are needed.

## Output format
Question: {{question}}
Reasoning:
Output:

## Few-shot (optional)
Example 1:
Question: <replace_with_example>
Reasoning: <replace_with_reasoning>
Output: 1

## Notes
- Use step-by-step reasoning before the final answer.
- The final line must be only 1, 2, or 3.
