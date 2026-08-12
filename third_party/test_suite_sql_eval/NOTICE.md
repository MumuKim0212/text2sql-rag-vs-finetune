Vendored from https://github.com/taoyds/test-suite-sql-eval @ e97acc546ecbee8fa27fa8dbf025ef61493a876c

License: Apache 2.0 (see LICENSE in this directory).

Files taken as-is: `parse.py`, `process_sql.py`.

Modification to `evaluation.py`: `evaluate()` upstream only prints results via
`print_scores()`; added `return scores` at the end so `rag_text2sql.eval` can
consume the scores dict programmatically instead of scraping stdout.

Modification to `exec_eval.py`: `exec_on_db_`/`exec_on_db` were `async def`, run
one `asyncio.run(...)` per query by `eval_exec_match`. Made them plain functions
called directly. The `asyncio.wait_for` timeout was unreachable -- `exec_on_db_`
has no `await`, so the coroutine never yields and the timeout callback cannot run
until after it finishes -- so this preserves semantics exactly while removing
~80k event loops per Spider dev run. On Windows each event loop builds its
self-pipe via `socket.socketpair()`, which falls back to a real TCP loopback
connection there; that `accept()` deadlocked a run partway through. Verified on a
150-example slice: identical exec and exact scores at every difficulty level,
before and after.

Not vendored (not needed for Spider/BIRD, only for the classical
ATIS/Academic/etc. datasets): `evaluate_classical.py`, `classical_test.pkl`,
`classical_provenance.ipynb`.
