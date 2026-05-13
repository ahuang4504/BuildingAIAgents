# Section 6: Discussion Questions

## 1. Before vs. After

What specific improvements did you observe after fine-tuning? Did the model learn SQL syntax, schema grounding (mapping column/table names correctly), or both?

- What was the change in accuracy on the 200 held-out test questions?
- How well did the fine-tuned model do on the additional manual test questions in Step 7?

> **Hint:** On the 200 in-distribution test questions, accuracy typically improves from ~37% (base model) to ~87% (fine-tuned). The Step 7 questions use novel schemas that were not in the training data.

The results show the model learned both SQL syntax and basic schema grounding well — it correctly used the right table/column names and produced valid SQL in most cases. However, it struggled with query logic on novel schemas:

> - **Easy cases**: Generated correct structure (`SELECT name ... WHERE department = 'engineering'`) but hallucinated extra filter conditions not mentioned in the question (e.g., `AND salary > 5000`, `AND category = 'food'`).
> - **Medium 1**: Applied `MAX()` to every selected column instead of just `score`, and added an unnecessary `GROUP BY`.
> - **Medium 2**: Got `GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3` right but included a non-aggregated `id` column in the `SELECT`.
> - **Hard (JOIN)**: Attempted a JOIN and used table aliases correctly, but fabricated a join condition on a column (`department`) that doesn't exist in `enrollments`, and grouped by course name instead of department.

Overall, the model improved significantly on the test set from 46% to 90% and was able to learn/understand basic SQL syntax and logic. The manual test questions posed a much harder problem though that the model still struggled with since they were novel schemas that the model had no context on and was unable to generalize to.

## 2. RAG Comparison

Imagine you had a RAG system with 1,000 (question, SQL) pairs in a vector database. For which of the Step 7 test questions above would RAG work well? For which would it struggle? Why?

RAG retrieves semantically similar examples and provides them as few-shot context. Since it directly injects retrieved text into the prompt, RAG would consequently work better for easy/medium cases where question phrases match the training examples so that the correct, relevant examples would be retrieved. RAG would struggle on the hard JOIN case since retrieving examples with patterns that involve multiple tables means that the question phrasing and the schema structure need to be similar to that in the external knowledge database or irrelevant confusing context will be retrieved, ultimately resulting in the wrong join logic.

## 3. Error Analysis

When the fine-tuned model gets a query wrong, how does it fail?

- **Wrong column names** — the model uses a plausible-sounding column that doesn't exist in the schema.
- **Wrong SQL syntax** — the generated SQL is not valid (missing clauses, wrong keywords, mismatched parentheses).
- **Wrong logic** — the SQL is syntactically valid but answers a different question (wrong aggregation, wrong filter, wrong join condition).

Each failure mode tells you something different about what the model did and did not learn during fine-tuning. Which failure mode is most common in your results?

In the Step 7 outputs, the dominant failure mode is **wrong logic**, not wrong syntax. Every generated query is syntactically valid SQL. The errors are semantic: hallucinating extra `WHERE` conditions that weren't asked for, applying `MAX()` to the wrong columns, or using a fabricated join condition (`T1.department = T2.department`) on a column that doesn't exist in `enrollments`. This suggests the fine-tune successfully taught SQL syntax and column-name grounding, but the model is still pattern-matching on query structure from training data rather than faithfully reasoning from the schema and question. Wrong-column errors are absent here because the schemas are simple and the model correctly identifies which columns to reference. The failure is in _how_ it uses them.
