"""
canonical_tactics.py — maps a Wazuh rule_id to ONE canonical MITRE tactic,
instead of trusting Wazuh's raw mitre.tactic field (which can list a
technique's FULL official tactic set — e.g. T1078/Valid Accounts expands
to Initial Access + Persistence + Privilege Escalation + Defense Evasion
on a single plain login). That expansion was shown to dominate naive
sequence-similarity scoring (see project log, 2026-10-08 finding).

This table is built from every rule ID actually observed in the lab so
far. On real (institute) data, alerts will fire rule IDs not in this
table — get_canonical_tactic() falls back to the first tactic in Wazuh's
own raw list in that case, AND logs the unmapped rule_id so the table can
be extended deliberately rather than silently guessing forever.
"""

# rule_id -> single canonical tactic, chosen from the lab's intended
# ground-truth meaning of that alert, not Wazuh's full auto-expanded list.
RULE_TO_TACTIC = {
    # Discovery
    # (nmap produces no Wazuh alerts at all — nothing to map here)

    # Credential Access — brute force / failed auth
    "5710": "Credential Access",   # sshd: nonexistent-user login attempt
    "5760": "Credential Access",   # sshd: authentication failed (real user)
    "5503": "Credential Access",   # PAM: user login failed
    "2502": "Credential Access",   # syslog: missed password more than once
    "100100": "Credential Access", # our custom brute-force seed rule

    # Initial Access — successful login
    "5715": "Initial Access",      # sshd: authentication success
    "5501": "Initial Access",      # PAM: login session opened
    "100101": "Initial Access",    # our custom "login after brute force" rule

    # Persistence
    "2832": "Persistence",         # syslog: crontab entry changed
    "100102": "Persistence",       # our custom "cron after login" rule

    # Defense Evasion / Privilege Escalation (unauthorized sudo attempt)
    "5405": "Defense Evasion",     # sudo: unauthorized user attempted sudo
    "100103": "Defense Evasion",   # our custom "full chain detected" rule

    # Collection
    "100110": "Collection",        # honeytoken file read (auditd)

    # Exfiltration
    "100111": "Exfiltration",      # scp execve (auditd)

    # Session bookkeeping — not attack-stage signal, excluded from sequences
    "5502": None,                  # PAM: login session closed
    "5402": None,                  # sudo: successful sudo to root (admin's own use, not attacker's)
}

_unmapped_seen = set()


def get_canonical_tactic(rule_id, raw_tactics_str):
    """
    Returns a single canonical tactic string, or None if this rule_id is
    known bookkeeping noise and should be excluded from the sequence.

    raw_tactics_str: the semicolon-joined mitre_tactics field from
    loader.py's joined CSV (used only as a fallback for unmapped rule_ids).
    """
    if rule_id in RULE_TO_TACTIC:
        return RULE_TO_TACTIC[rule_id]

    # Unmapped rule_id (expected on real/institute data) — fall back to
    # the first raw tactic Wazuh gave us, but flag it so the table can be
    # deliberately extended rather than silently trusting the expansion.
    if rule_id not in _unmapped_seen:
        _unmapped_seen.add(rule_id)
        first = raw_tactics_str.split(";")[0] if raw_tactics_str else "(none)"
        print(f"[canonical_tactics] UNMAPPED rule_id={rule_id!r} -> "
              f"falling back to first raw tactic {first!r}. "
              f"Add a real mapping to RULE_TO_TACTIC if this recurs.")
    if not raw_tactics_str:
        return None
    return raw_tactics_str.split(";")[0]


def unmapped_rule_ids():
    """Call after processing a batch to see every rule_id that had no
    explicit mapping and used the fallback — use this to extend the table."""
    return sorted(_unmapped_seen)
