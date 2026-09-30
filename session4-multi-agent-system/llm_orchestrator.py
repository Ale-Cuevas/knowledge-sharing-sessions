import importlib
import re
# import os
# import sqlite3

from typing import Callable
from dataclasses import dataclass, field


ma = importlib.import_module("multi-agent")

# to remember what has happened, store the state
@dataclass
class RunState:
    question: str  # the question
    tables: list[str] | None = None  # 
    sql: str | None = None
    rows: list[dict] | None = None
    last_error: str | None = None
    answer: str | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    attempts_counter: int = 0


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[RunState], str]


class LLMOrchestrator:
    def __init__(self, llm, conn):
        # TODO: create agents, build self.tools
        self.conn = conn
        self.schema_agent = ma.SchemaAgent(llm)
        self.sql_agent = ma.SQLAgent(llm)
        self.answer_agent = ma.AnswerAgent(llm)
        self.columns = ma.get_columns(conn)
        self.foreign_keys = ma.get_foreign_keys(conn)
        self.tools: dict[str, Tool] = {
            t.name: t for t in [Tool("select_tables",
                                     "Pick the database tables relevant to the question",
                                     self._tool_select_tables)]
        }

    def _tool_select_tables(self, state: RunState) -> str:
        # 1. read input: the question that comes from state, the schema from db
        full_schema = ma.get_schema(self.conn)
        
        # 2. do the work: schemaAgent picks tables, python adds the join tables
        tables = self.schema_agent.run(state.question)
        if tables is None:
            # 3. write state
            return "No table holds this information. The question is out of scope."
        tables = self.add_join_tables(tables)
        
        # 3. write state
        state.tables = tables
        
        # 4. observation: what happened and what changed, in one line
        return f"Selected tables: {', '.join(tables)}"

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
