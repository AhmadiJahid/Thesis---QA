You are classifying question complexity.

Rules:
- First decide if the question requires shared constraints or multiple related entities.
- If YES, output 3.
- Otherwise, decide if an intermediate entity is required.
  - If YES, output 2.
  - If NO, output 1.

Output ONLY one number: 1, 2, or 3.
Do not explain your answer.

Examples:

Q: What genre is Inception?
A: 1

Q: Who directed movies starring Brad Pitt?
A: 2

Q: Who acted in movies whose directors also directed Inception?
A: 3

Q: {question}
A: 
