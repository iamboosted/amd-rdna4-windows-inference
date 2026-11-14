"""Validate the public evidence bundle using the Python standard library only."""
import argparse
import copy
import csv
import getpass
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import statistics

FILES = {'README.md', 'METHODOLOGY.md', 'ROCM-ROADMAP.md', 'PRIVACY.md', 'verify.py',
         'data/results.json', 'data/runs.csv', 'data/provenance.json', 'SHA256SUMS',
         'data/lfm-results.json', 'qwen-9060xt/README.md', 'qwen-9060xt/results.json'}
CLEAN = ['control-01', 'optimized-01', 'optimized-02', 'control-02']
PHASES = {'verify_sync', 'feature_submit', 'sampling_commit', 'draft', 'other'}
PATTERNS = {
    'absolute machine path': r'(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]|/(?:home|Users)/[^\s/]+|\\\\[A-Za-z0-9_.-]+\\',
    'email address': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'credential-like value': r'(?i)(?:api[_-]?key|authorization|password|secret)\s*[=:]\s*[\x22\x27]?[A-Za-z0-9_./+=-]{12,}',
    'access token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b',
    'hardware identifier': r'(?i)(?:PCI|USB)\\(?:VEN_|VID_)|S-1-5-21-[0-9-]+',
    'URL credentials': r'https?://[^\s/]+@',
}


def privacy_categories(text, private_terms):
    found = {label for label, pattern in PATTERNS.items() if re.search(pattern, text)}
    for candidate in re.findall(r'(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])', text):
        try:
            ipaddress.ip_address(candidate)
            found.add('network address')
        except ValueError:
            pass
    if any(re.search(r'(?i)(?<![\w])' + re.escape(term) + r'(?![\w])', text)
           for term in private_terms if term):
        found.add('private identity term')
    return found


def require(condition, label):
    if not condition:
        raise ValueError(label)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def validate_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            validate_numbers(item)
    elif isinstance(value, list):
        for item in value:
            validate_numbers(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'nonfinite numeric evidence')


def validate_data(data):
    validate_numbers(data)
    require(data['schema_version'] == 1, 'schema version')
    require(data['clean_run_order'] == CLEAN, 'clean run order')
    require(len(data['runs']) == 6, 'run inventory')
    by_run = {r['run']: r for r in data['runs']}
    require(set(by_run) == set(CLEAN + ['audit-01', 'profile-01']), 'run labels')
    for label, row in by_run.items():
        require(row['prompt_tokens'] == 4570 and row['output_tokens'] == 3109, 'token accounting')
        require(row['accepted_proposals'] == 2038 and row['proposals'] == 3213 and row['rounds'] == 1071, 'proposal accounting')
        require(row['accepted_proposals'] + row['rounds'] == row['output_tokens'], 'emitted token reconciliation')
        require(row['tests_passed'] == row['tests_total'] == 25, 'functional checks')
        for key in ('thinking', 'token_ids_identical', 'reasoning_identical', 'answer_identical', 'code_identical', 'owned_cleanup_confirmed'):
            require(row[key] is True, 'correctness or cleanup flag')
        require(row['guard_trip'] is False, 'guard outcome')
        require(row['instrumented'] is (label not in CLEAN), 'diagnostic separation')
        require(row['server_rate_counted_tokens'] == row['output_tokens'] - 1, 'server rate convention')
        require(close(row['decode_tok_s'], row['server_rate_counted_tokens'] * 1000 / row['decode_ms']), 'server decode rate')
        require(close(row['emitted_tokens_per_decode_s'], row['output_tokens'] * 1000 / row['decode_ms']), 'emitted-token decode rate')
        require(set(row['host_per_round_ms']) == PHASES, 'phase inventory')
        require(close(sum(row['host_per_round_ms'].values()), row['decode_ms']/row['rounds']), 'host timing reconciliation')
        require(row['dispatch']['selected'] == 3213 and row['dispatch']['cycle'] == 1071, 'dispatch coverage')
        require(row['dispatch']['custom'] == (0 if label.startswith('control') else 3213), 'replacement coverage')
    rates = {}
    for mode in ('control', 'optimized'):
        rows = [by_run[label] for label in CLEAN if label.startswith(mode)]
        rates[mode] = sum(r['output_tokens'] for r in rows)*1000/sum(r['decode_ms'] for r in rows)
        require(close(rates[mode], data['headline'][mode + '_pooled_decode_tok_s']), 'pooled rate')
        require(close(rates[mode]*3108/3109, data['headline'][mode + '_pooled_server_convention_tok_s']), 'pooled server convention')
        for phase in PHASES:
            mean = sum(r['host_per_round_ms'][phase] for r in rows)/len(rows)
            require(close(mean, data['mean_host_phases_ms'][mode][phase]), 'phase mean')
    require(close(data['headline']['gain_percent'], 100*(rates['optimized']/rates['control']-1)), 'gain calculation')
    require(close(data['headline']['gain_tok_s'], rates['optimized']-rates['control']), 'absolute gain')
    require(close(data['headline']['tokens_per_round'], 3109/1071), 'tokens per round')
    require(close(data['headline']['acceptance_percent'], 100*2038/3213), 'acceptance rate')
    needed = sum(data['mean_host_phases_ms']['optimized'].values()) - 1000*(3109/1071)/100
    require(close(needed, data['headline']['additional_ms_per_round_for_100_tok_s']), '100 token target arithmetic')
    require(close(needed+10/1071, data['headline']['additional_ms_per_round_for_100_server_convention_tok_s']), '100 token server-convention arithmetic')
    audit = data['numerical_audit']
    require(audit['scores_compared'] == 3213*248144 == 797286672, 'score audit coverage')
    require(all(audit[k] == 0 for k in ('bit_mismatches', 'nonfinite', 'guard_corruptions')), 'score audit outcome')
    require(by_run['audit-01']['dispatch']['compared_scores'] == audit['scores_compared'], 'audit run reconciliation')
    profile = data['profile']
    require(close(profile['markov_group_speed_ratio'], profile['original_markov_ms_per_three_calls']/profile['optimized_markov_ms_per_three_calls']), 'profile ratio')
    require(data['baseline_fixture'] == {'upstream_tests': 17, 'upstream_passed': 17, 'regression_tests': 8, 'regression_passed': 4}, 'baseline test evidence')


def self_test(data):
    samples = [
        ('absolute machine path', 'Z:' + chr(92) + 'private' + chr(92) + 'note'),
        ('network address', '.'.join(('192', '0', '2', '1'))),
        ('email address', 'test' + '@' + 'example.invalid'),
        ('credential-like value', 'api' + '_key=' + 'A'*24),
        ('private identity term', 'private' + '_identity_sample'),
        ('hardware identifier', 'PCI' + chr(92) + 'VEN_' + '1234'),
    ]
    terms = ['private' + '_identity_sample']
    for kind, text in samples:
        require(kind in privacy_categories(text, terms), 'privacy self-test')
    require(not privacy_categories('Q8_0, gfx1200, 93.536 tok/s; https://github.com/ggml-org/llama.cpp', terms), 'false positive self-test')
    mutants = []
    x = copy.deepcopy(data); x['headline']['gain_percent'] += 1; mutants.append(x)
    x = copy.deepcopy(data); x['runs'][1]['token_ids_identical'] = False; mutants.append(x)
    x = copy.deepcopy(data); x['numerical_audit']['scores_compared'] -= 1; mutants.append(x)
    x = copy.deepcopy(data); x['runs'][1]['instrumented'] = True; mutants.append(x)
    for mutant in mutants:
        try:
            validate_data(mutant)
        except (ValueError, KeyError, ZeroDivisionError):
            continue
        raise ValueError('invalid evidence accepted in self-test')
    return len(samples) + 1 + len(mutants)


def validate_lfm(data):
    validate_numbers(data)
    require(data['single_gpu_inference'] is True and data['gpu'] == 'RX 9060 XT', 'LFM GPU scope')
    require(data['no_ngram'] is True and data['no_custom_kernel'] is True, 'LFM method')
    control, draft = data['runs']
    require([r['mode'] for r in data['runs']] == ['control', 'dspark'], 'LFM run inventory')
    for row in data['runs']:
        require(row['context_capacity'] == 65536 and row['prompt_tokens'] == 4705, 'LFM context scope')
        require(row['kv'] == row['draft_kv'] == 'q8_0/q8_0', 'LFM KV precision')
        require(row['server_counted_tokens'] == row['output_tokens'] - 1, 'LFM token convention')
        require(close(row['decode_tok_s'], row['server_counted_tokens']*1000/row['decode_ms']), 'LFM decode rate')
        require(close(row['prefill_tok_s'], row['prompt_tokens']*1000/row['prefill_ms']), 'LFM prefill rate')
        require(row['upstream_passed'] == 17 and row['regression_passed'] == 8, 'LFM grading')
        require(row['fix_works'] and row['thinking_present'] and row['cleanup_confirmed'], 'LFM completion flags')
        require(row['finish'] == 'stop' and row['output_tokens'] < 16384, 'LFM natural stop')
    require(control['output_tokens'] == 9326 and draft['output_tokens'] == 10177, 'LFM different lengths')
    require(close(data['decode_gain_percent'], 100*(draft['decode_tok_s']/control['decode_tok_s']-1)), 'LFM speed gain')
    require(close(data['request_time_reduction_percent'], 100*(1-draft['request_seconds']/control['request_seconds'])), 'LFM request time')
    s = draft['speculation']
    require(s['proposals'] == s['rounds']*9 == 15048, 'LFM proposal counts')
    require(s['accepted_proposals'] + s['rounds'] + 1 == draft['output_tokens'], 'LFM emitted accounting')
    require(len(data['artifacts']) == 6, 'LFM provenance coverage')
    for artifact in data['artifacts']:
        require(set(artifact) == {'file', 'sha256'}, 'LFM artifact allowlist')
        require('/' not in artifact['file'] and chr(92) not in artifact['file'], 'LFM artifact basename')
        require(re.fullmatch('[0-9a-f]{64}', artifact['sha256']) is not None, 'LFM artifact hash')


def validate_qwen(data):
    validate_numbers(data)
    c = data['configuration']
    require(data['single_gpu_inference'] is True and c['gpu'] == 'RX 9060 XT', 'Qwen GPU scope')
    require(c['vocabulary'] == 248320 and c['proposals'] == 7, 'Qwen full vocabulary and proposals')
    require(c['context_capacity'] == 8192 and c['target_kv'] == 'q8_0/q8_0' and c['draft_kv'] == 'q4_0/q4_0', 'Qwen original cache scope')
    by_run = {r['run']: r for r in data['runs']}
    require(len(by_run) == len(data['runs']) == 12, 'Qwen selected run inventory')
    for r in data['runs']:
        require(r['measurement_mode'] == 'clean' and r['graphs_disabled'] is False, 'Qwen clean measurement')
        require(close(r['aggregate_decode_tok_s'], r['output_tokens']*1000/r['total_decode_ms']), 'Qwen pooled rate')
        require(sum(t['passed'] for t in r['task_results']) == r['cases_passed'], 'Qwen task score sum')
        require(sum(t['cases'] for t in r['task_results']) == r['cases'] == 167, 'Qwen test coverage')
        require(sum(t['passed'] == t['cases'] for t in r['task_results']) == r['tasks_passed'], 'Qwen task passes')
        for filename, digest in r['runtime_sha256'].items():
            require('/' not in filename and chr(92) not in filename and re.fullmatch('[0-9a-f]{64}', digest), 'Qwen artifact identity')
    for label in ['draft-mmq-control', 'draft-mmq-enabled', 'draft-mmq-final', 'draft-mmq-final-repeat']:
        r = by_run[label]
        require(r['exact_token_streams_vs_original'] == [True]*6 and r['output_tokens'] == 4069, 'Qwen exact outputs')
        require(r['cases_passed'] == 155 and r['tasks_passed'] == 5, 'Qwen retained errors')
    off, on = by_run['draft-mmq-control'], by_run['draft-mmq-enabled']
    require(off['runtime_sha256'] == on['runtime_sha256'], 'Qwen same-build toggle')
    before, after = data['power_comparison']['runs']
    for r in [before, after]:
        require(close(r['median_tps'], statistics.median(r['per_task_tps'].values())), 'Qwen power median')
        require(close(r['coverage_percent'], 100*r['covered_request_seconds']/r['total_request_seconds']), 'Qwen power coverage')
        require(r['accuracy'] == {'tokens': 4069, 'cases_passed': 155, 'cases': 167, 'tasks_passed': 5, 'exact_task_token_matches': 6}, 'Qwen power accuracy')
    p = data['power_comparison']
    require(close(p['median_tps_change_percent'], 100*(after['median_tps']/before['median_tps']-1)), 'Qwen power throughput change')
    require(close(p['average_reported_power_change_percent'], 100*(after['average_adlx_power_w']/before['average_adlx_power_w']-1)), 'Qwen reported power change')
    f = data['larger_context_followup']
    require(f['context_capacity'] == 16384 and f['target_kv'] == f['draft_kv'] == 'q8_0/q8_0', 'Qwen later cache scope')
    require(f['prompt_tokens'] == 10258 and f['output_tokens'] == 2608, 'Qwen longer prompt')
    require(close(f['decode_tok_s'], (f['output_tokens']-1)*1000/f['decode_ms']), 'Qwen follow-up rate')
    require(f['raw_project_output_accepted'] is False and f['formatting_only_diagnostic'] is True, 'Qwen formatting limitation')
    require(f['diagnostic_checks_passed'] == 875 and f['diagnostic_workflows_passed'] == 39, 'Qwen diagnostic limitation')
    require(f['short_tasks']['cases_passed'] == 132 and f['short_tasks']['tasks_passed'] == 4, 'Qwen context-task limitation')
    require(f['measured_at_32k'] is False and f['guard_trip'] is False, 'Qwen context and guard limits')
    a = data['acceptance']
    require(a['accepted_tokens'] + a['rounds'] + a['initial_tokens_outside_speculative_rounds'] == a['output_tokens'] == 4069, 'Qwen accepted accounting')
    require(close(a['ms_per_round'], a['decode_ms']/a['rounds']), 'Qwen round timing')
    require(close(a['round_emitted_tokens_mean'], (a['output_tokens']-6)/a['rounds']), 'Qwen tokens per round')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deny-term', action='append', default=[])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '.git' not in p.relative_to(root).parts}
    require(actual == FILES, 'unexpected or missing publication files')
    terms = list(args.deny_term)
    for value in (getpass.getuser(), os.environ.get('COMPUTERNAME', ''), os.environ.get('HOSTNAME', '')):
        if len(value) >= 4:
            terms.append(value)
    texts = {}
    for name in sorted(FILES):
        path = root/name
        require(not path.is_symlink(), 'symbolic link in publication')
        texts[name] = path.read_text(encoding='utf-8')
        categories = privacy_categories(texts[name], terms)
        require(not categories, name + ': ' + ', '.join(sorted(categories)))
    manifest = {}
    for line in texts['SHA256SUMS'].splitlines():
        content_hash, name = line.split('  ', 1)
        require(name in FILES - {'SHA256SUMS'} and name not in manifest, 'manifest inventory')
        require(re.fullmatch('[0-9a-f]{64}', content_hash) is not None, 'manifest digest')
        manifest[name] = content_hash
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == content_hash, name + ': checksum mismatch')
    require(set(manifest) == FILES - {'SHA256SUMS'}, 'incomplete manifest')
    data = json.loads(texts['data/results.json'])
    validate_data(data)
    lfm = json.loads(texts['data/lfm-results.json'])
    qwen = json.loads(texts['qwen-9060xt/results.json'])
    validate_lfm(lfm)
    validate_qwen(qwen)
    by_run = {r['run']: r for r in data['runs']}
    with (root/'data/runs.csv').open(newline='', encoding='utf-8') as f:
        csv_rows = list(csv.DictReader(f))
    require([r['run'] for r in csv_rows] == CLEAN, 'CSV order')
    for row in csv_rows:
        source = by_run[row['run']]
        for key in ('decode_tok_s', 'emitted_tokens_per_decode_s', 'server_rate_counted_tokens', 'decode_ms', 'output_tokens', 'tests_passed', 'tests_total'):
            require(close(row[key], source[key]), 'CSV numeric agreement')
        for column, phase in (('draft_ms_per_round', 'draft'), ('verify_sync_ms_per_round', 'verify_sync')):
            require(close(row[column], source['host_per_round_ms'][phase]), 'CSV phase agreement')
        require(row['token_ids_identical'] == 'True', 'CSV exactness')
    provenance = json.loads(texts['data/provenance.json'])
    require(provenance['runtime_commit'] == data['configuration']['runtime_commit'], 'runtime provenance')
    for artifact in provenance['artifacts']:
        require(set(artifact) == {'role', 'file', 'sha256'}, 'artifact allowlist')
        require(re.fullmatch('[0-9a-f]{64}', artifact['sha256']) is not None, 'artifact identity')
        require('/' not in artifact['file'] and chr(92) not in artifact['file'], 'artifact basename')
    for name, text in texts.items():
        if name.endswith('.md'):
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                if target.startswith(('https://', '#')):
                    continue
                require((root/name).parent.joinpath(target.split('#', 1)[0]).is_file(), name + ': broken relative link')
    checks = self_test(data) if args.self_test else 0
    if args.self_test:
        mutants = [(validate_lfm, copy.deepcopy(lfm)), (validate_qwen, copy.deepcopy(qwen))]
        mutants[0][1]['runs'][1]['decode_tok_s'] = 200
        mutants[1][1]['larger_context_followup']['raw_project_output_accepted'] = True
        for validate, mutant in mutants:
            try:
                validate(mutant)
            except ValueError:
                checks += 1
            else:
                raise ValueError('Invalid added report evidence accepted')
    print(json.dumps({'status': 'pass', 'publication_files': len(FILES), 'hashes_verified': len(manifest),
                      'clean_runs': len(CLEAN), 'evidence_arithmetic': 'pass', 'privacy_scan': 'pass',
                      'self_tests': checks}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, AssertionError, OSError) as error:
        # Input data values are deliberately not echoed on failure.
        print('Validation failed: ' + (str(error) if isinstance(error, ValueError) else type(error).__name__))
        raise SystemExit(1)
