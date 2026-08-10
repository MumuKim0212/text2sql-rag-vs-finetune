Vendored from https://github.com/taoyds/test-suite-sql-eval @ e97acc546ecbee8fa27fa8dbf025ef61493a876c

License: Apache 2.0 (see LICENSE in this directory).

Files taken as-is: `exec_eval.py`, `parse.py`, `process_sql.py`.

Modification to `evaluation.py`: `evaluate()` upstream only prints results via
`print_scores()`; added `return scores` at the end so `rag_text2sql.eval` can
consume the scores dict programmatically instead of scraping stdout.

Not vendored (not needed for Spider/BIRD, only for the classical
ATIS/Academic/etc. datasets): `evaluate_classical.py`, `classical_test.pkl`,
`classical_provenance.ipynb`.
