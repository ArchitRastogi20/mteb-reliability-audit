## MMTEB Filter-View Demonstration

A standalone reproduction script (`analysis/reproduce_filter_view.py`) loads the public CSV and demonstrates the filter behavior on `bge-m3`: the BelebeleRetrieval score under the Italian filter (N/A) differs from the global-task-list score (78.16) because the filter changes which tasks are averaged, not which language within a task is read. The per-language CSV for the Italian filter exposes only `eng-Latn` -- no `ita-Latn` column exists.
