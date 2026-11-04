# Runbook: High CPU Utilization

Symptoms: sustained CPU above 90%, rising request latency, autoscaler churn.

Triage steps:
1. Identify the hot process with `top` / container metrics and confirm which service is saturating CPU.
2. Check for a recent deploy or traffic spike that correlates with the increase.
3. Scale out the service horizontally to shed load while you investigate.
4. If a single request pattern is responsible, add caching or rate limiting for that endpoint.
5. Roll back the most recent deploy if the regression started right after it.
