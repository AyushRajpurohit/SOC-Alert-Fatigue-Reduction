"""Local-only Wazuh Dev Tools response sanitizer for SOC alert-fatigue evaluation.

Input: saved JSON response from a restricted _source Wazuh Indexer search.
Output: pseudonymized Wazuh JSON lines compatible with correlation_engine.py,
        and a NON-identifying aggregate summary.

Do NOT upload raw source, secret key, or event-level sanitized output without
explicit approval and human privacy review. Pseudonymization != anonymization.
"""
import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import stat
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

RULE_RE = re.compile(r"^[0-9]{1,12}$")
TECH_RE = re.compile(r"^T[0-9]{4}(?:\.[0-9]{3})?$")


def get_key(path):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement and key rotation.
        with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:
            f.write(secrets.token_bytes(32))
    data = p.read_bytes()
    if len(data) != 32:
        raise ValueError('Key must be exactly 32 bytes')
    if os.name == 'posix' and p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError('Secret key permissions too broad; use chmod 600 on a filesystem supporting Unix modes')
    return data


def token(key, domain, value, length=18):
    digest = hmac.new(key, (domain + '\0' + str(value).strip().lower()).encode(), hashlib.sha256).hexdigest()
    return digest[:length]


def private_ip_alias(key, value, aliases):
    if value in (None, ''):
        return None
    try:
        ip = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None
    normalized = ip.compressed
    if normalized in aliases:
        return aliases[normalized]
    digest = hmac.new(key, ('ip\0' + normalized).encode(), hashlib.sha256).digest()
    if ip.version == 4:
        # Benchmark-only IPv4 address block, not an institution endpoint.
        x = int.from_bytes(digest[:4], 'big') % 131072
        alias = str(ipaddress.IPv4Address(int(ipaddress.IPv4Address('198.18.0.0')) + x))
    else:
        # IPv6 documentation prefix.
        x = int.from_bytes(digest[:12], 'big')
        alias = str(ipaddress.IPv6Address(int(ipaddress.IPv6Address('2001:db8::')) + x))
    if alias in aliases.values():
        raise ValueError('Pseudonym address collision: abort, do not silently merge entities')
    aliases[normalized] = alias
    return alias


def parse_timestamp(value, shift_days):
    if not isinstance(value, str):
        raise ValueError('Missing timestamp')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Naive timestamp disallowed')
    return (dt.astimezone(timezone.utc) - timedelta(days=shift_days)).isoformat()


def scalar(v):
    return v if isinstance(v, (str, int)) and not isinstance(v, bool) else None


def sanitize_hit(hit, key, aliases, shift_days, index):
    s = hit.get('_source', {})
    if not isinstance(s, dict):
        raise ValueError('Missing _source object')
    agent = s.get('agent') or {}
    data = s.get('data') or {}
    rule = s.get('rule') or {}
    if not all(isinstance(v, dict) for v in (agent, data, rule)):
        raise ValueError('Agent, data, rule must be objects')
    rule_id = str(scalar(rule.get('id')) or '')
    if not RULE_RE.fullmatch(rule_id):
        raise ValueError('Invalid or missing numeric rule ID')
    try:
        level = int(rule.get('level'))
    except (ValueError, TypeError):
        raise ValueError('Invalid severity')
    if not 0 <= level <= 20:
        raise ValueError('Out-of-range severity')
    agent_id = scalar(agent.get('id'))
    if not agent_id:
        raise ValueError('Missing agent ID')
    mitre = rule.get('mitre') or {}
    if not isinstance(mitre, dict):
        mitre = {}
    ids = mitre.get('id') or []
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list):
        ids = []
    techniques = sorted(set(v.upper() for v in ids if isinstance(v, str) and TECH_RE.fullmatch(v.upper())))
    clean_data = {}
    for field in ('srcip', 'dstip'):
        alias = private_ip_alias(key, data.get(field), aliases)
        if alias:
            clean_data[field] = alias
    username = scalar(data.get('dstuser'))
    if username:
        clean_data['dstuser'] = 'user_' + token(key, 'user', username)
    output = {
        'id': f'event_{index:08d}',
        'timestamp': parse_timestamp(s.get('timestamp'), shift_days),
        'agent': {'id': 'host_' + token(key, 'agent', agent_id)},
        'rule': {'id': rule_id, 'level': level},
    }
    if techniques:
        output['rule']['mitre'] = {'id': techniques}
    if clean_data:
        output['data'] = clean_data
    return output


def run(raw, sanitized, summary, key_file, shift_days=365):
    paths = [Path(raw).resolve(), Path(sanitized).resolve(), Path(summary).resolve(), Path(key_file).resolve()]
    if len(set(paths)) != 4:
        raise ValueError('Input, outputs, and key must be four distinct paths')
    if Path(sanitized).exists() or Path(summary).exists():
        raise FileExistsError('Refusing to overwrite existing output')
    source = json.loads(Path(raw).read_text(encoding='utf-8'))
    hits = source.get('hits', {}).get('hits')
    if not isinstance(hits, list):
        raise ValueError('Expected Dev Tools response with hits.hits list')
    if not hits:
        raise ValueError('No hit records in response')
    key = get_key(key_file)
    aliases = {}
    rows = [sanitize_hit(hit, key, aliases, shift_days, i) for i, hit in enumerate(hits, 1)]
    # Strict schema check prevents accidental field leaks.
    for item in rows:
        assert set(item) <= {'id', 'timestamp', 'agent', 'rule', 'data'}
        assert set(item['agent']) == {'id'}
        assert set(item['rule']) <= {'id', 'level', 'mitre'}
        if 'data' in item:
            assert set(item['data']) <= {'srcip', 'dstip', 'dstuser'}
    counts = Counter(r['rule']['id'] for r in rows)
    summary_obj = {
        'source_type': 'institute_Wazuh_approved_local_sample',
        'records': len(rows),
        'distinct_rules': len(counts),
        'distinct_agents': len({r['agent']['id'] for r in rows}),
        'with_mitre': sum('mitre' in r['rule'] for r in rows),
        'with_source_ip': sum('srcip' in r.get('data', {}) for r in rows),
        'with_destination_ip': sum('dstip' in r.get('data', {}) for r in rows),
        'with_destination_user': sum('dstuser' in r.get('data', {}) for r in rows),
        'severity_counts': {str(k): v for k, v in sorted(Counter(r['rule']['level'] for r in rows).items())},
        'notes': ['No incident labels; these counts do not imply benign/malicious status.',
                  'Source is a limited paginated sample, not a full one-hour export.',
                  'Pseudonymized event-level file remains sensitive and is not intended for sharing.']
    }
    Path(sanitized).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(sanitized, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    with os.fdopen(os.open(summary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', encoding='utf-8') as f:
        json.dump(summary_obj, f, indent=2)
    return summary_obj


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Locally pseudonymize approved Wazuh Dev Tools exports')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--key-file', required=True, help='Private stable key: never share or commit')
    p.add_argument('--shift-days', type=int, default=365)
    a = p.parse_args()
    report = run(a.input, a.output, a.summary, a.key_file, a.shift_days)
    print(json.dumps(report, indent=2))
