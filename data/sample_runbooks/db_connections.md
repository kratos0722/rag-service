# Runbook: Database Connection Pool Exhausted

Symptoms: "too many connections" errors, timeouts acquiring a connection, API 5xx spikes.

Triage steps:
1. Inspect current connections and identify which service is holding the most.
2. Kill idle-in-transaction sessions that are leaking connections.
3. Increase the connection pool size or add a pooler such as PgBouncer if demand is real.
4. Look for a missing connection close in a recently deployed code path.
5. Set a connection saturation alert to page before the pool is fully exhausted.
