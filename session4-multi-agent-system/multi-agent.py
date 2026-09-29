"""
Minimal multi-agent NL -> SQL workflow, running a free Hugging Face model locally.

Same architecture as the Bedrock version:
  Orchestrator  -> plain Python that owns the state and decides who runs next
  SchemaAgent   -> picks only the tables needed (keeps context small)
  SQLAgent      -> writes the query (and gets error feedback on failure)
  run_query     -> the "tool"; in your repo this is what the MCP server exposes
  AnswerAgent   -> turns rows into a natural-language answer

See README.md for a diagram of how these pieces interact.

Setup (no AWS account needed; the model downloads once, ~3 GB):
  uv sync                      # or: pip install torch transformers sentencepiece langchain-core
  uv run python multi-agent.py

Try a different small model without changing any code:
  HF_MODEL_ID=Qwen/Qwen2.5-Coder-0.5B-Instruct uv run python multi-agent.py   # faster, but fails on joins
  HF_MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct uv run python multi-agent.py     # slower, more robust
"""
import os
import re
import sqlite3

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Model to load from the Hugging Face Hub; override with the HF_MODEL_ID env var.
# 1.5B is the smallest Qwen coder model that handles joins; 0.5B invents columns
# (e.g. bookings.price) and can't tell when a question is out of scope.
MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
# Number of *extra* attempts the SQL agent gets after its first query fails
# (so the total number of attempts is MAX_RETRIES + 1).
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# 1. A toy database (stands in for your internal DB)
# ---------------------------------------------------------------------------
def build_demo_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database filled with sample data.

    The database has four tables: ``customers``, ``flights``, ``bookings`` and
    ``employees``. ``employees`` is deliberately unrelated to the others so we
    can see whether the SchemaAgent correctly leaves it out.

    Returns:
        An open ``sqlite3.Connection`` to the in-memory database. The data
        disappears when the connection is closed.
    """
    # ":memory:" means the DB lives only in RAM - nothing is written to disk.
    conn = sqlite3.connect(":memory:")
    # executescript runs several ";"-separated SQL statements in one call.
    # The REFERENCES clauses declare foreign keys: get_schema() shows them to
    # the LLM so it knows how the tables join.
    conn.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE flights   (flight_id INTEGER PRIMARY KEY, origin TEXT, destination TEXT, price REAL);
        CREATE TABLE bookings  (booking_id INTEGER PRIMARY KEY,
                                customer_id INTEGER REFERENCES customers(customer_id),
                                flight_id INTEGER REFERENCES flights(flight_id), booked_on TEXT);
        CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name TEXT, department TEXT);

        INSERT INTO customers VALUES (1,'Ana','Mexico'),(2,'Luis','Chile'),(3,'Sofia','Mexico');
        INSERT INTO flights   VALUES (10,'MEX','MAD',850),(11,'SCL','LIM',320),(12,'MEX','CUN',120);
        INSERT INTO bookings  VALUES (100,1,10,'2026-01-05'),(101,1,12,'2026-02-11'),
                                     (102,2,11,'2026-02-20'),(103,3,12,'2026-03-01');
        INSERT INTO employees VALUES (1,'Marta','Sales'),(2,'Pedro','IT');
    """)
    return conn


# One-line description per table, shown to the LLM next to the table definition.
# In a real system this would come from your data catalog / documentation.
TABLE_DESCRIPTIONS = {
    "customers": "people who book flights",
    "flights": "flights and their ticket price",
    "bookings": "which customer booked which flight",
    "employees": "company staff",
}


def get_schema(conn: sqlite3.Connection) -> dict[str, str]:
    """Describe each table as a one-line ``CREATE TABLE`` statement.

    Example output::

        {"bookings": "CREATE TABLE bookings (booking_id INTEGER PRIMARY KEY, "
                     "customer_id INTEGER REFERENCES customers(customer_id), ...); "
                     "-- which customer booked which flight", ...}

    Code models (Qwen-Coder, etc.) are trained on text-to-SQL datasets that
    present the schema as DDL, so this format - with types, foreign keys and a
    short description - works much better than a bare ``table(col1, col2)``
    list. The foreign keys matter most: they tell the model that e.g. a
    booking's price must be looked up in ``flights``.

    Args:
        conn: Open connection to the database to inspect.

    Returns:
        A dict mapping each table name to its one-line description.
    """
    schema = {}
    for t in get_columns(conn):
        # PRAGMA table_info: one row per column (cid, name, type, notnull, default, pk).
        # PRAGMA foreign_key_list: one row per FK (id, seq, table, from, to, ...).
        fks = {fk[3]: f"{fk[2]}({fk[4]})" for fk in conn.execute(f"PRAGMA foreign_key_list({t})")}
        cols = []
        for _, name, col_type, _, _, pk in conn.execute(f"PRAGMA table_info({t})"):
            col = f"{name} {col_type}" + (" PRIMARY KEY" if pk else "")
            if name in fks:
                col += f" REFERENCES {fks[name]}"
            cols.append(col)
        comment = f" -- {TABLE_DESCRIPTIONS[t]}" if t in TABLE_DESCRIPTIONS else ""
        schema[t] = f"CREATE TABLE {t} ({', '.join(cols)});{comment}"
    return schema


def get_columns(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return the column names of every table, e.g. ``{"flights": ["flight_id", ...]}``.

    Used by the Orchestrator to turn a raw SQLite error into a precise hint.
    """
    # sqlite_master is SQLite's internal catalog; each row describes one object
    # (table, index, view...). r[0] is the "name" column we selected.
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    # PRAGMA table_info returns one row per column:
    # (cid, name, type, notnull, default_value, pk) -> c[1] is the column name.
    return {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] for t in tables}


def get_foreign_keys(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Return which tables each table references, e.g. ``{"bookings": {"customers", "flights"}}``."""
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    # PRAGMA foreign_key_list returns one row per FK: (id, seq, table, from, to, ...)
    # -> fk[2] is the referenced table.
    return {t: {fk[2] for fk in conn.execute(f"PRAGMA foreign_key_list({t})")} for t in tables}


# ---------------------------------------------------------------------------
# 2. The tool (what an MCP server would expose as e.g. `run_query`)
# ---------------------------------------------------------------------------
def run_query(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Execute a read-only SQL query and return the rows as dictionaries.

    This is the only function that touches the database on behalf of the
    agents - it is the "tool" in the architecture. In a production setup this
    would be exposed by an MCP server instead of being called directly.

    Args:
        conn: Open connection to the database.
        sql: The SQL query produced by the SQLAgent.

    Returns:
        Up to 50 rows, each as a ``{column_name: value}`` dict.

    Raises:
        ValueError: If the query is not a SELECT (read-only guardrail).
        sqlite3.Error: If SQLite rejects the query (syntax error, unknown
            column, ...). The Orchestrator catches this and feeds the message
            back to the SQLAgent so it can fix the query.
    """
    # Guardrail: never let LLM-generated SQL modify data (INSERT/UPDATE/DROP...).
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")  # read-only guardrail
    cur = conn.execute(sql)
    # cur.description is a tuple per result column; c[0] is the column name.
    cols = [c[0] for c in cur.description]
    # zip pairs each column name with its value -> {"name": "Ana", "country": "Mexico"}.
    # fetchmany(50) caps the result size so a huge table can't flood the LLM prompt.
    return [dict(zip(cols, row)) for row in cur.fetchmany(50)]


# ---------------------------------------------------------------------------
# 3. The LLM: a local Hugging Face model wrapped as a LangChain Runnable
# ---------------------------------------------------------------------------
def make_llm(model_id: str = MODEL_ID) -> RunnableLambda:
    """Load a Hugging Face model locally and wrap it as a LangChain Runnable.

    Works with both model families:
      * encoder-decoder (seq2seq) models such as T5 / flan-t5, and
      * decoder-only (causal / GPT-style) models such as Qwen.

    Wrapping the model in a ``RunnableLambda`` lets the agents compose it with
    the pipe syntax ``prompt | llm | parser`` exactly as they would with a
    hosted model (e.g. Bedrock or OpenAI).

    Args:
        model_id: Hugging Face Hub id of the model to load.

    Returns:
        A Runnable that takes a LangChain prompt value and returns the
        generated text as a string.
    """
    print(f"Loading {model_id} (first run downloads it)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Read only the config (cheap) to decide which model class to instantiate.
    is_seq2seq = AutoConfig.from_pretrained(model_id).is_encoder_decoder  # T5 = True, GPT-style = False
    model_cls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        pretrained_model_name_or_path=model_id)
    model.eval()  # inference mode: disables dropout and other training-only behavior

    def generate(prompt_value) -> str:
        """Run one generation step: prompt in, model text out.

        Args:
            prompt_value: The object produced by a ``PromptTemplate`` (a
                LangChain ``PromptValue``) with all variables filled in.

        Returns:
            The newly generated text, without the prompt and special tokens.
        """
        text = prompt_value.to_string()
        if not is_seq2seq and tokenizer.chat_template:  # instruct models expect chat formatting
            # Wrap the plain prompt in the model's chat markup, e.g.
            # "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n".
            # add_generation_prompt=True appends the "assistant" header so the
            # model knows it is its turn to speak.
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        # Convert text -> token ids as PyTorch tensors ("pt"). Anything past
        # 1024 tokens is cut off so a runaway prompt can't blow up memory
        # (the DDL schema + question use ~250).
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        # no_grad: we are not training, so skip gradient bookkeeping (faster, less memory).
        with torch.no_grad():
            # do_sample=False -> greedy decoding: always pick the most likely next
            # token, so the same prompt gives the same answer every run.
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        if not is_seq2seq:  # causal models echo the prompt; keep only the new tokens
            # output has shape [batch, prompt_len + new_len]; slice off the first
            # prompt_len tokens of every row so only the generated part remains.
            output = output[:, inputs["input_ids"].shape[1]:]
        # output[0] = first (and only) sequence in the batch; token ids -> text.
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Wrapping in RunnableLambda keeps the same `prompt | llm | parser` syntax as before
    return RunnableLambda(generate)


# ---------------------------------------------------------------------------
# 4. The agents: each one = focused prompt + parsing of its output.
#    Small local models can't do reliable structured (JSON) output,
#    so each agent parses plain text defensively instead.
# ---------------------------------------------------------------------------
class SchemaAgent:
    """Selects which database tables are relevant to a question.

    Sending only the relevant tables to the SQLAgent keeps its prompt short
    (important for small models) and reduces the chance it joins the wrong
    tables. It can also decide that *no* table is relevant, which lets the
    Orchestrator refuse out-of-scope questions instead of hallucinating SQL.
    """

    def __init__(self, llm):
        """Build the agent's chain: prompt -> LLM -> plain string.

        Args:
            llm: A LangChain Runnable that turns a prompt into text (see ``make_llm``).
        """
        # We ask for *columns* (table.column), not tables: small models answer
        # "which tables?" by listing almost everything, but naming the exact
        # columns forces them to find where each piece of data lives
        # (e.g. "amount spent" -> flights.price, not a made-up bookings.price).
        prompt = PromptTemplate.from_template(
            "{schema}\n\n"
            "Which columns (as table.column) are needed to answer the question below? "
            "Reply with a comma-separated list only. "
            "If no column above holds the information, reply NONE.\n"
            "Question: {question}"
        )
        # The "|" operator chains Runnables: the output of each step is the
        # input of the next (LangChain Expression Language, LCEL).
        self.chain = prompt | llm | StrOutputParser()

    def run(self, question: str, schema: dict[str, str]) -> list[str] | None:
        """Ask the LLM which columns are needed and turn them into table names.

        Args:
            question: The user's natural-language question.
            schema: Full schema as returned by ``get_schema``.

        Returns:
            Names of the tables the model mentioned that really exist in the
            schema, or ``None`` if the model says the database can't answer
            the question. If the answer is unusable (no real table and no
            "NONE"), every table is returned as a safe fallback.
        """
        raw = self.chain.invoke({"question": question, "schema": "\n".join(schema.values())})
        print(f"[schema agent] raw output: {raw!r}")
        # Don't trust the model's formatting: instead of parsing its answer,
        # check which real table names appear in it. \b = word boundary, so
        # "flights" is not matched inside e.g. "flightsXYZ". This also drops
        # any hallucinated table names.
        tables = [t for t in schema if re.search(rf"\b{t}\b", raw, re.IGNORECASE)]
        if tables:
            return tables
        if re.search(r"\bNONE\b", raw):
            return None  # the model says the data isn't in this database
        return list(schema)  # unusable answer -> fall back to all tables


class SQLAgent:
    """Translates a natural-language question into a SQLite SELECT query.

    When a previous attempt failed, the Orchestrator passes the error message
    back as ``feedback`` so the model can correct its query (self-correction loop).
    """

    def __init__(self, llm):
        """Build the agent's chain: prompt -> LLM -> plain string.

        Args:
            llm: A LangChain Runnable that turns a prompt into text (see ``make_llm``).
        """
        # This "DDL + SQL comment + SELECT" layout mirrors the text-to-SQL
        # data code models were trained on, so they reply with bare SQL
        # instead of an explanation.
        prompt = PromptTemplate.from_template(
            "{schema}\n\n"
            "-- Using valid SQLite, answer the following question for the tables provided above.\n"
            "-- Use only the columns listed above.\n"
            "{feedback}"  # empty on the first try; error message on retries
            "-- {question}\n"
            "SELECT"
        )
        self.chain = prompt | llm | StrOutputParser()

    def run(self, question: str, schema: str, feedback: str = "") -> str:
        """Generate a SQL query for the question.

        Args:
            question: The user's natural-language question.
            schema: ``CREATE TABLE`` statements of only the relevant tables
                (the SchemaAgent's selection).
            feedback: SQL comment lines describing the previous failed
                attempt, or ``""`` on the first attempt.

        Returns:
            The extracted SQL query string (not yet validated or executed).
        """
        raw = self.chain.invoke({"question": question, "schema": schema, "feedback": feedback})
        return self._extract_sql(raw)

    @staticmethod
    def _extract_sql(raw: str) -> str:
        """Pull the first SQL statement out of the model's free-form answer.

        Models often wrap SQL in Markdown fences, add a sentence before it
        ("Here is the SELECT query:") or an explanation after it; this strips
        all of that.

        Args:
            raw: The model's raw text output.

        Returns:
            A single query starting with ``SELECT``, without the trailing ``;``.
            If no query is found, the raw text is returned (and ``run_query``
            will reject it).
        """
        # If there is a ```sql ... ``` block, the query is inside it.
        fenced = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL)
        if fenced:
            raw = fenced.group(1)
        # The prompt ends with "SELECT", so the model may continue from there
        # without repeating it (e.g. "name FROM customers ...").
        if not re.match(r"\s*select\b", raw, re.IGNORECASE):
            if not re.search(r"^\s*select\b", raw, re.IGNORECASE | re.MULTILINE):
                raw = "SELECT " + raw
        # Start at the first SELECT that begins a line (skips prose such as
        # "Here is the SELECT query:"), and stop at the first ";" so any
        # explanation after the query is dropped.
        # MULTILINE: "^" matches at every line start; DOTALL: "." spans lines.
        match = re.search(r"^\s*(select\b[^;]*)", raw, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else raw.strip()


class AnswerAgent:
    """Turns raw database rows into a one-sentence natural-language answer."""

    def __init__(self, llm):
        """Build the agent's chain: prompt -> LLM -> plain string.

        Args:
            llm: A LangChain Runnable that turns a prompt into text (see ``make_llm``).
        """
        prompt = PromptTemplate.from_template(
            "Question: {question}\n"
            "Database results: {rows}\n"
            "Answer the question in one sentence using only the database results. "
            "Mention every result row."
        )
        self.chain = prompt | llm | StrOutputParser()

    def run(self, question: str, rows: list[dict]) -> str:
        """Write the final answer for the user.

        Args:
            question: The user's natural-language question.
            rows: Query results returned by ``run_query``.

        Returns:
            A natural-language answer. If there are no rows, a fixed message is
            returned without calling the LLM (avoids a hallucinated answer).
        """
        if not rows:
            return "The query returned no results."
        # Only send the first 10 rows to keep the prompt within the model's context.
        return self.chain.invoke({"question": question, "rows": rows[:10]})


# ---------------------------------------------------------------------------
# 5. The orchestrator: owns the state, routes work, handles retries
# ---------------------------------------------------------------------------
class Orchestrator:
    """Coordinates the agents and the tool to answer one question end to end.

    The orchestrator is plain Python, not an LLM: the control flow
    (schema -> SQL -> tool -> answer, with retries) is fixed and predictable.
    Only the individual steps use the model. Anything that can be checked
    deterministically (join tables, which table owns a column) is done here
    in code rather than trusted to the model.
    """

    OUT_OF_SCOPE = "The database doesn't contain information to answer that question."

    def __init__(self, llm, conn):
        """Create the three agents, all sharing the same LLM.

        Args:
            llm: The LangChain Runnable used by every agent (see ``make_llm``).
            conn: Database connection the tool will query.
        """
        self.conn = conn
        self.schema_agent = SchemaAgent(llm)
        self.sql_agent = SQLAgent(llm)
        self.answer_agent = AnswerAgent(llm)
        self.columns = get_columns(conn)
        self.foreign_keys = get_foreign_keys(conn)

    def add_join_tables(self, tables: list[str]) -> list[str]:
        """Add any table that links two of the selected tables.

        E.g. for ``["customers", "flights"]`` this adds ``bookings``, because
        bookings references both. Without it the SQLAgent can't write the join
        and tends to invent a shortcut column instead.
        """
        bridges = [t for t, refs in self.foreign_keys.items()
                   if t not in tables and len(refs & set(tables)) >= 2]
        return tables + bridges

    def explain_error(self, error: Exception) -> str:
        """Turn a SQLite error into a hint the SQLAgent can act on.

        The raw message ("no such column: b.price") says what is wrong but not
        how to fix it; small models then repeat the same query. Here we look
        up where the column really lives and say so explicitly.
        """
        msg = str(error)
        if m := re.search(r"no such column: (?:\w+\.)?(\w+)", msg):
            col = m.group(1)
            owners = [t for t, cols in self.columns.items() if col in cols]
            if owners:
                return f"{msg}. Column {col} exists only in table(s): {', '.join(owners)}"
            return f"{msg}. No table has a column named {col}"
        if "no such table" in msg:
            return f"{msg}. Available tables: {', '.join(self.columns)}"
        return msg

    def ask(self, question: str) -> str:
        """Answer a natural-language question about the database.

        Pipeline:
            1. SchemaAgent picks the relevant tables (or says none are).
            2. SQLAgent writes a query for those tables.
            3. ``run_query`` executes it; on error, a hint is fed back to
               the SQLAgent and it tries again (up to ``MAX_RETRIES`` retries).
            4. AnswerAgent turns the rows into a sentence.

        Args:
            question: The user's natural-language question.

        Returns:
            The final answer, or an explanation if the question can't be answered.
        """
        print(f"\n=== Question: {question}")

        # Step 1: schema agent narrows the context
        full_schema = get_schema(self.conn)
        tables = self.schema_agent.run(question, full_schema)
        if tables is None:
            # Stop here: letting the SQLAgent try anyway only produces
            # hallucinated tables and columns.
            return self.OUT_OF_SCOPE
        tables = self.add_join_tables(tables)
        print(f"[schema agent] tables: {tables}")

        # Step 2 + 3: SQL agent writes, tool executes, errors loop back
        feedback = ""
        # range(1, MAX_RETRIES + 2) -> attempts 1..MAX_RETRIES+1 (first try + retries).
        for attempt in range(1, MAX_RETRIES + 2):
            sub_schema = "\n".join(full_schema[t] for t in tables)
            sql = self.sql_agent.run(question, sub_schema, feedback)
            print(f"[sql agent] attempt {attempt}: {sql}")
            try:
                rows = run_query(self.conn, sql)
                print(f"[tool] {len(rows)} row(s): {rows}")
                break  # success -> leave the loop (and skip the "else" below)
            except Exception as e:
                hint = self.explain_error(e)
                print(f"[tool] error: {hint}")
                # If the query needed a column from a table the SchemaAgent
                # left out, widen the schema so the next attempt can use it.
                missing = [t for t in self.columns if t not in tables and re.search(rf"\b{t}\b", hint)]
                if missing and not hint.startswith("no such table"):
                    tables = self.add_join_tables(tables + missing)
                    print(f"[orchestrator] added tables: {tables}")
                # Put the failing query and the hint in the next prompt (as SQL
                # comments, matching the prompt format) so the model can see
                # what went wrong and correct it.
                one_line_sql = " ".join(sql.split())
                feedback = f"-- The query {one_line_sql} failed: {hint}. Write a corrected query.\n"
        else:
            # for/else: this block runs only if the loop finished WITHOUT "break",
            # i.e. every attempt failed.
            return "I couldn't build a working query for that question."

        # Step 4: answer agent writes the final response
        return self.answer_agent.run(question, rows)


if __name__ == "__main__":
    # Load the model once and share it between all agents.
    orchestrator = Orchestrator(make_llm(), build_demo_db())
    for q in [
        "What are the names of the customers from Mexico?",  # single table + filter
        "Which flights cost more than 300?",                 # single table + numeric filter
        "What is the total amount spent by each customer?",  # needs a 3-table join + aggregation
        "the number of horses in each stable?",              # a question without answer
    ]:
        print("[answer]", orchestrator.ask(q))
