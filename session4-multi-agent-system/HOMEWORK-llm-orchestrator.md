# 📘 Homework: From Workflow to Agent: Building an LLM Orchestrator

**Course:** Knowledge Sharing Sessions, Session 4 (Multi-Agent Systems)
**Estimated time:** 4–6 hours
**Prerequisites:** You have run `multi-agent.py` and you understand what each agent does.

## Learning objectives
When you finish, you should be able to:
1. Explain the difference between a **workflow**, where code picks the next step, and an
   **agent**, where the LLM picks the next step, and name the trade-offs of each.
2. Describe a set of tools to an LLM so that it can choose between them.
3. Implement the **agent loop**: prompt, then decision, then parse, then execute, then
   observe, and repeat.
4. Parse LLM output defensively and recover from bad decisions.
5. Add guardrails (step budgets, preconditions, loop detection) that stop an agent from
   running away.
6. Test an LLM-driven system without an LLM by using a scripted fake model.

## Background reading (≈30 min)
- Anthropic, *"Building effective agents"*, especially the sections on "workflows vs agents"
  and on "orchestrator-workers".
- Yao et al. 2022, *"ReAct: Synergizing Reasoning and Acting in Language Models"*. Reading the
  abstract and Figure 1 is enough.
- Re-read the docstring of `Orchestrator` in `multi-agent.py`. It says *"The orchestrator is
  plain Python, not an LLM"*. In this homework you remove that assumption.

---

## Part 0: Concepts (answer in writing before coding)

The current `Orchestrator.ask()` is a **workflow**. The order of the steps is fixed in Python:

```
SchemaAgent → add_join_tables → SQLAgent → run_query ─┬─ ok ──→ AnswerAgent → answer
                                   ↑                  │
                                   └─ explain_error ←─┘ (up to MAX_RETRIES)
```

An **LLM orchestrator** replaces the arrows with a question that the LLM answers at every step:
*"Given what has happened so far, which tool should run next?"*

```
          ┌──────────────────────────────────────────────┐
          ▼                                              │
  build prompt (question + tools + history)              │
          │                                              │
     LLM decides ── "Action: write_sql"                  │
          │                                              │
   parse + validate ── invalid? → observation: "error"  ─┤
          │                                              │
   execute tool (Python) → observation: "3 rows: ..."   ─┘
          │
   Action: finish → return answer
```

**Key idea: the LLM decides and Python acts.** The LLM never touches the database or the
state directly. It only chooses a name from a menu. Your code owns the state, runs the tool,
and reports what happened back to the LLM.

**Q0.1** List two things the fixed workflow does better than an LLM orchestrator, and two
things an LLM orchestrator could do that the workflow cannot. (Hint: think about
predictability and cost on one side, and about questions that need two queries or none on
the other.)

**Q0.2** Look at `SchemaAgent.run`, `SQLAgent._extract_sql` and `explain_error`. Which of them
are "deterministic helpers" that should **stay in Python** even when an LLM orchestrates?
Why?

---

## Part 1: Setup

Create `session4-multi-agent-system/llm_orchestrator.py`. The file `multi-agent.py` has a
hyphen in its name, so a normal `import` does not work. Use `importlib` instead:

```python
import importlib
ma = importlib.import_module("multi-agent")   # safe: its demo is behind if __name__ == "__main__"

# Now you can use ma.make_llm, ma.build_demo_db, ma.SchemaAgent, ma.SQLAgent,
# ma.AnswerAgent, ma.run_query, ma.get_schema, ma.get_columns, ma.get_foreign_keys
```

✅ **Checkpoint:** `uv run python -c "import importlib; importlib.import_module('multi-agent')"`
runs without errors and does not load the model.

---

## Part 2: The state (the orchestrator's memory)

Because the LLM decides the order of steps, your code has to remember everything that has
happened so far. Write a small `@dataclass` called `RunState`.

```python
from dataclasses import dataclass, field

@dataclass
class RunState:
    question: str
    tables: list[str] | None = None      # set by select_tables
    sql: str | None = None               # set by write_sql
    rows: list[dict] | None = None       # set by run_query (on success)
    last_error: str | None = None        # set by run_query (on failure), used as feedback
    answer: str | None = None            # set by write_answer
    history: list[tuple[str, str]] = field(default_factory=list)  # (action, observation)
    # TODO: think about what else you need, e.g. a counter of SQL attempts
```

**Q2.1** Why is `sql` stored in the state instead of the LLM copying the SQL into its next
decision? (Hint: think about what a 1.5B model does when it has to repeat 200 characters
exactly.)

---

## Part 3: Tools (the menu the LLM chooses from)

Wrap every capability as a **tool**. A tool has a name, a short description for the LLM,
and a Python function that reads and updates `RunState` and returns an **observation**
string.

| Tool name        | What it does (Python side)                                                 | Precondition |
|------------------|----------------------------------------------------------------------------|--------------|
| `select_tables`  | calls `SchemaAgent.run`, then `add_join_tables`; stores `state.tables`     | none |
| `write_sql`      | calls `SQLAgent.run` with the sub-schema and `feedback` built from `state.last_error` | `tables` is set |
| `run_query`      | calls `run_query(conn, state.sql)`; on success sets `rows`, on failure sets `last_error` (use your `explain_error`) | `sql` is set |
| `write_answer`   | calls `AnswerAgent.run(question, rows)`; stores `state.answer`             | `rows` is set |
| `finish`         | ends the loop and returns `state.answer`, or the out-of-scope message      | none |

Suggested structure:

```python
@dataclass
class Tool:
    name: str
    description: str                      # ONE line, written for the LLM
    run: Callable[[RunState], str]        # returns the observation text

class LLMOrchestrator:
    def __init__(self, llm, conn):
        # TODO: create the three agents (reuse ma.SchemaAgent etc.)
        # TODO: build self.tools: dict[str, Tool]
        ...

    def _tool_select_tables(self, state: RunState) -> str: ...
    def _tool_write_sql(self, state: RunState) -> str: ...
    # ...
```

**Rules for writing good observations** (these matter more than you might expect):
- Keep them **short**, one or two lines. The prompt gets longer at every step (see Part 5).
- Say what happened **and** what state changed, for example `"Selected tables: customers,
  bookings, flights"` or `"Query failed: no such column: b.price. Column price exists only
  in table(s): flights"`.
- Show at most about three rows, followed by `"... (N rows total)"`.
- If `SchemaAgent` returns `None`, the observation should say so plainly, for example
  `"No table holds this information."`. The LLM then has to decide to `finish`.

**Reuse:** Move `add_join_tables` and `explain_error` over from the old `Orchestrator`. You
can copy them or call them through a small helper. Do not ask the LLM to do their work.

✅ **Checkpoint:** Call each tool function by hand, in the correct order, on a fresh
`RunState` for "Which flights cost more than 300?". You should get rows and an answer without
any orchestrator LLM involved.

---

## Part 4: The orchestrator prompt

This is the most important design work in the homework. The prompt must contain:
1. **Role:** one sentence explaining that the model coordinates tools to answer a question
   about a database.
2. **The tools:** one line per tool, `name: description`, generated from `self.tools` and
   not hardcoded.
3. **The question.**
4. **The history:** the numbered `(action, observation)` pairs so far.
5. **The output format:** exactly what to reply, and nothing else.

Starter template (improve it):

```
You coordinate tools to answer a question about a SQL database.

Tools:
{tool_list}

Question: {question}

Steps so far:
{history}          <- "(none yet)" on the first step

Choose the next tool. Reply with exactly one line:
Action: <tool name>
```

Tips for small models:
- Ending the prompt with `Action:` works like the `SELECT` trick in `SQLAgent`: the model
  only has to complete the tool name.
- Keep the tool descriptions **action-oriented**, for example "Write a SQL query for the
  selected tables". They should not read like documentation.
- Add a one-line hint about what a usual order looks like, but do not hardcode it. Watch how
  much the hint changes the model's behavior (Experiment E2).

**Q4.1** Where is the line between "giving the LLM a hint" and "secretly writing a workflow
inside the prompt"? Take a position and defend it.

---

## Part 5: Parsing the decision

Write `parse_action(raw: str, tool_names: list[str]) -> str | None`.

Requirements (write a unit test for **each** of them):

| Raw model output                              | Expected |
|-----------------------------------------------|----------|
| `"Action: write_sql"`                         | `"write_sql"` |
| `" write_sql"` (the prompt ended with `Action:`) | `"write_sql"` |
| `"I think we should run_query now."`          | `"run_query"` |
| `"Action: Write_SQL"`                         | `"write_sql"` |
| `"Action: execute"`                           | `None` |
| `"select_tables then write_sql"`              | `"select_tables"` (the **first** match wins) |

Hint: use the technique `SchemaAgent.run` already uses, which is to search for **known**
names instead of trusting the format. Keep the `\b` word boundaries.

---

## Part 6: The agent loop

Implement `LLMOrchestrator.ask(question) -> str`:

```python
def ask(self, question: str) -> str:
    state = RunState(question)
    for step in range(1, MAX_STEPS + 1):
        prompt = self._build_prompt(state)
        raw = self.chain.invoke(...)          # orchestrator's own prompt | llm | StrOutputParser()
        action = parse_action(raw, list(self.tools))
        print(f"[orchestrator] step {step}: {raw!r} -> {action}")

        # TODO 1: action is None       -> observation "Unknown action. Choose one of: ..."
        # TODO 2: precondition fails   -> observation explaining what must happen first
        # TODO 3: action == "finish"   -> return the answer (or OUT_OF_SCOPE)
        # TODO 4: otherwise run the tool, append (action, observation) to state.history
    # TODO 5: step budget exhausted -> return a graceful failure message
```

**Important design decision:** When the LLM picks an invalid action, **do not raise an
exception**. Turn the mistake into an **observation** and let the LLM try again. This is the
same self-correction idea as the SQL retry loop, applied one level higher.

⚠️ **Gotcha: context truncation.** `make_llm` truncates prompts at **1024 tokens**, and it
cuts from the **end**. That means the instruction `Action:` is the first thing to disappear
when the history grows. Print `len(tokenizer(prompt).input_ids)` at every step, or keep
observations short enough that you never reach the limit. **Q6.1:** Why is truncating from
the right the worst case for an agent loop?

---

## Part 7: Guardrails

An LLM in control can loop forever or skip steps. Add all of the following:

1. **Step budget:** `MAX_STEPS = 8`.
2. **Preconditions:** use the table in Part 3. A violation becomes an observation, not a
   crash.
3. **SQL attempt budget:** after `MAX_RETRIES + 1` failed `run_query` calls, remove
   `write_sql` from the menu, or make it return "No attempts left, finish.".
4. **Loop detection:** if the same action with the same resulting state runs twice in a
   row, add a nudge to the observation. For example, if `select_tables` runs twice, the
   observation should say "Tables already selected. Next, write SQL."
5. **Finish without an answer:** if `finish` is chosen while `state.answer` is `None`, decide
   what should happen. If the tables are `None`, return the out-of-scope message. Otherwise,
   either refuse to finish or return a fallback message. Document your choice.

**Q7.1** Guardrails 2 and 4 move some control back into Python. Is your system still an
"LLM orchestrator"? Where on the workflow–agent spectrum does it now sit?

---

## Part 8: Test without the LLM

Loading a 3 GB model to test the loop logic is slow and non-deterministic. Build a **fake
LLM** instead:

```python
from langchain_core.runnables import RunnableLambda

def scripted_llm(replies: list[str]) -> RunnableLambda:
    it = iter(replies)
    return RunnableLambda(lambda prompt_value: next(it))
```

The orchestrator and the three agents share one LLM, so the script must include **every**
call in order: orchestrator decisions **and** agent outputs. Alternatively, you can give
the orchestrator its own LLM parameter, `orchestrator_llm=`, so that you only have to script
the orchestrator's decisions and can pass the real agents a separate fake. **That design
choice is part of the exercise, and a good answer explains it.**

Write at least these tests (plain `assert` in a `tests_llm_orchestrator.py`, or pytest):
- **T1 happy path:** select_tables, write_sql, run_query, write_answer, finish. The test
  returns the answer.
- **T2 out of scope:** select_tables returns `None`, then finish. The test returns
  `OUT_OF_SCOPE`.
- **T3 bad action:** the LLM replies `"banana"` once and then behaves correctly. The loop
  recovers.
- **T4 precondition:** the LLM picks `run_query` first. The observation mentions that SQL
  must be written first.
- **T5 runaway:** the LLM always says `select_tables`. The loop stops at `MAX_STEPS` with the
  failure message.
- **T6 SQL retry:** the first SQL is invalid and the second is valid. The test checks that
  `last_error` was passed to `SQLAgent` as feedback.

---

## Part 9: Run it for real

Change the `__main__` block to run **both** orchestrators on the same four questions that
`multi-agent.py` uses. Record the results in a table:

| Question | Old: answer ok? | New: answer ok? | New: # steps | New: # LLM calls | Time (s) |
|----------|-----------------|-----------------|--------------|------------------|----------|

Experiments (write two or three sentences about each):
- **E1 Model size:** run with the default 1.5B model, then with
  `HF_MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct`. How often does the orchestrator make an
  invalid choice with each?
- **E2 Order hint:** remove the "usual order" hint from the prompt. What breaks?
- **E3 New question type:** ask "How many customers are there, and how many flights?" Can
  the old orchestrator answer it? Can yours? What would your tools need so that the new
  orchestrator could run **two** queries?

---

## Bonus (pick at least one)
- **B1 Tools with arguments:** let the orchestrator write `Action: write_sql | hint: use a
  JOIN`, and pass the hint to `SQLAgent` as extra feedback. Update `parse_action`.
- **B2 Native tool calling:** Qwen2.5's chat template accepts
  `tokenizer.apply_chat_template(messages, tools=[...])` and replies with
  `<tool_call>{...}</tool_call>` JSON. Build a second `make_llm` variant that uses it and
  compare how reliable it is with your text protocol.
- **B3 Hybrid:** keep the fixed workflow and call the LLM orchestrator only when the workflow
  fails. This is a common production pattern. Measure the cost and accuracy.

---

## Deliverables and grading

| Item | Points |
|------|--------|
| Written answers Q0.1, Q0.2, Q2.1, Q4.1, Q6.1, Q7.1 | 15 |
| `RunState` + tools with good observations (Part 2–3) | 15 |
| Prompt builder + `parse_action` with its unit tests (Part 4–5) | 15 |
| Agent loop (Part 6) | 20 |
| Guardrails (Part 7) | 15 |
| Fake-LLM tests T1–T6 all passing (Part 8) | 10 |
| Comparison table + experiments E1–E3 (Part 9) | 10 |
| Bonus | +10 |

**Rules of the course:** You may copy helper functions from `multi-agent.py`. Do not copy
code from an agent framework such as LangGraph or `create_react_agent`. The goal is to
build the loop yourself. Once your version works, comparing it to LangGraph is a good
follow-up.

**Getting help:** When you get stuck, ask Claude for a *hint* on a specific part, for
example "hint for Part 7 loop detection". Hints come in levels (nudge, then approach, then
pseudo-code) and do not include solution code unless you ask for it.

