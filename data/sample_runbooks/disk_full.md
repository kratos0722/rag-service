# Runbook: Disk Space Full

Symptoms: disk usage near 100%, failed writes, log rotation errors on the host.

Triage steps:
1. Find the largest directories with `du -sh` and confirm what is consuming the volume.
2. Rotate and compress old logs, and clear temporary files that are safe to delete.
3. Expand the attached volume if the growth is legitimate and will continue.
4. Check for runaway log output from a crashing process writing to disk.
5. Add a disk-usage alert threshold so this is caught earlier next time.
