"""Four-group live HTTP evaluation. No replacement of Port B or skill results."""
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(os.environ.get('LIVE_EVAL_ROOT', 'E:/DeepTumour_agent_eval/live_portb_v1'))
DATA_ROOT = Path(os.environ.get('LIVE_EVAL_DATA_ROOT', str(ROOT)))
load_dotenv('//wsl.localhost/Ubuntu-22.04/home/voxelsage/VoxelSage/.env')
os.environ['PORT_B_INTERNAL'] = 'http://127.0.0.1:8875'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
os.environ['no_proxy'] = os.environ['NO_PROXY']
os.environ['CACHE_ROOT'] = str(ROOT / 'agent_sessions')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core.server as server


class Sink:
    async def send_json(self, payload):
        pass


def dump(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def score(answer, expected, unit, tol):
    try:
        value = json.loads(re.findall(r'\{[^{}]*"value"\s*:[^{}]*\}', answer)[-1])
    except (ValueError, IndexError):
        return {'format_valid': False, 'answer_correct': False}
    actual = value.get('value')
    if isinstance(expected, bool) or expected is None:
        ok = actual is expected
    elif isinstance(expected, str):
        ok = actual == expected
    else:
        ok = type(actual) in (int, float) and math.isfinite(actual) and abs(actual - expected) <= tol
    return {'format_valid': True, 'answer_correct': ok and value.get('unit') == unit}


def summarize(rows):
    summary = {}
    for kind in ['cold', 'cache', 'missing', 'multi']:
        rs = [r for r in rows if r['kind'] == kind]
        summary[kind] = {'n': len(rs), 'answer_correct': sum(r.get('score', {}).get('answer_correct', False) for r in rs),
                         'exceptions': sum('error' in r for r in rs),
                         'no_additional_calls': sum(not r.get('calls') for r in rs),
                         'median_seconds': statistics.median(r['seconds'] for r in rs) if rs else None}
    dump('smoke_summary.json', summary)


async def main():
    assert not server.llm_configuration_errors()
    skills = await server.port_b_list_skills(force_refresh=True)
    assert len(skills['skills']) == 8
    cases = {r['case']: r for r in json.loads((DATA_ROOT / 'live_cases.json').read_text(encoding='utf-8'))}
    questions = [q for q in json.loads((DATA_ROOT / 'original_questions_smoke.json').read_text(encoding='utf-8')) if q['image_id'] in cases]
    units = {'liver lesion existence': ('bool', 0), 'lesion counting': ('count', 0),
             'organ volume measurement': ('cm3', .051), 'lesion volume measurement': ('cm3', .051),
             'largest lesion diameter': ('cm', .051)}
    manifest_path=DATA_ROOT/'manifest.json'
    agreement=json.loads(manifest_path.read_text(encoding='utf-8')).get('answer_agreement_tolerances',{}) if manifest_path.exists() else {}
    tasks = []
    for q in questions:
        unit, tol = units[q['question_subtype']]
        expected = q['answer'].lower() == 'yes' if unit == 'bool' else float(q['answer'])
        rule=agreement.get(q['question_subtype'],{})
        if isinstance(expected,(int,float)) and not isinstance(expected,bool) and 'relative' in rule:
            tol=max(rule['absolute_floor'],abs(expected)*rule['relative'])
        tasks.append({'id': q['qid'], 'cases': [cases[q['image_id']]['port_b_case']], 'query': q['question'],
                      'expected': expected, 'unit': unit, 'tolerance': tol, 'subtype': q['question_subtype'],
                      'source_case': q['image_id'], 'official_answer': q['answer']})
    dump('smoke_tasks.json', tasks)
    formal = manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')).get('protocol') == 'live_portb_formal_v1'
    dump('smoke_runtime.json', {'model': server.LLM_MODEL_NAME, 'port_b': server.PORT_B_INTERNAL,
         'port_b_execution': 'real HTTP API, real SkillEngine computation, no monkey patch',
         'max_rounds': 4, 'max_new_tokens': 700, 'timeout': 180, 'concurrency': 2,
         'scope': 'same-grid reference-mask measurements; CT-mask geometry validated before inclusion' if formal else 'intrinsic reference-mask measurements only; CT-mask overlay not validated',
         'group_5': 'withdrawn', 'stage': 'formal first batch' if formal else 'smoke, not full large-cohort replacement',
         'code_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(server.__file__)]}})
    result_path = ROOT/'smoke_results.jsonl'
    rows = [json.loads(line) for line in result_path.read_text(encoding='utf-8').splitlines()] if result_path.exists() else []
    assert len({r['id'] for r in rows}) == len(rows), 'Duplicate result IDs in resumable output'
    existing = {r['id']: r for r in rows}
    sem = asyncio.Semaphore(2)

    def session(t):
        s = server.create_empty_session('live_' + t['id'])
        s.update({'case_ids': t['cases'], '_active_case_ids': t['cases'], '_active_volumes': [c+'.nii.gz' for c in t['cases']],
                  'available_skills': skills['skills'], 'available_tools': server.inject_target_case_id(skills['tools'],t['cases']),
                  'max_new_tokens':700, 'segmentation_status': 'completed'})
        for cid in t['cases']:
            s['tool_store'][cid] = {'segmentation': {'status': 'reference_masks', 'case_id': cid}}
        return s

    async def execute(t, kind, s=None):
        result_id = kind+'_'+t['id']
        if result_id in existing:
            return s, existing[result_id]
        async with sem:
            s = s if s is not None else session(t)
            s['agent_round'] = 0
            s['skill_call_history'] = []
            s['original_question'] = t['query']
            s['current_user_query'] = t['query'] + '\nAnswer concisely. End with one JSON object: {"value": <number, true, false, null if unavailable, or case ID for comparison>, "unit": "' + t['unit'] + '"}. Do not guess missing evidence.'
            started = time.monotonic()
            try:
                answer = await asyncio.wait_for(server.run_agent_loop(Sink(), s, max_rounds=4),timeout=180)
                r = {'id': kind+'_'+t['id'], 'kind': kind, 'cases': t['cases'], 'answer': answer,
                     'score': score(answer,t['expected'],t['unit'],t['tolerance']),
                     'calls': list(s['skill_call_history']), 'rounds': s.get('agent_round')}
                s['conversation'].extend([{'role':'user','content':t['query']},{'role':'assistant','content':answer}])
            except Exception as e:
                r = {'id':kind+'_'+t['id'],'kind':kind,'cases':t['cases'],'error':type(e).__name__, 'calls':list(s['skill_call_history'])}
            r['seconds'] = round(time.monotonic()-started,3)
            rows.append(r)
            existing[r['id']] = r
            with (ROOT/'smoke_results.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
            summarize(rows)
            print('PROGRESS',len(rows),r['id'],r.get('score'),flush=True)
            return s, r

    async def cold_and_cache(t):
        prior = existing.get('cold_'+t['id'])
        if prior:
            s = session(t)
            for call in prior.get('calls', []):
                if call.get('status') == 'ok' and call.get('result') is not None:
                    cid = call.get('params', {}).get('target_case_id') or t['cases'][0]
                    s['tool_store'].setdefault(cid, {})[call.get('skill_name')] = call['result']
            if prior.get('answer'):
                s['conversation'].extend([{'role':'user','content':t['query']},{'role':'assistant','content':prior['answer']}])
            r = prior
        else:
            s, r = await execute(t,'cold')
        # Same actual session, including real skill cache and conversation.
        await execute(t,'cache',s)
    await asyncio.gather(*(cold_and_cache(t) for t in tasks))
    await asyncio.gather(*(execute(dict(t, cases=[cases[t['source_case']]['missing_case']],expected=None),'missing') for t in tasks))
    ids = sorted(cases)
    volume_by_case = {r['case']: r['masks']['liver'].get('volume_cm3') for r in json.loads((DATA_ROOT/'geometry_audit.json').read_text(encoding='utf-8')) if r['case'] in cases}
    if any(v is None for v in volume_by_case.values()):
        import nibabel as nib, numpy as np
        for cid in cases:
            p=DATA_ROOT/'source_cases'/cid/'liver.nii.gz'; m=nib.load(p)
            volume_by_case[cid]=float(np.count_nonzero(np.asanyarray(m.dataobj))*abs(np.linalg.det(m.affine[:3,:3]))/1000)
    for i,a in enumerate(ids):
        b=ids[(i+1)%len(ids)]
        aid,bid=cases[a]['port_b_case'],cases[b]['port_b_case']
        t={'id':'pair_'+str(i),'cases':[aid,bid], 'query':f'Compare liver volumes of {aid} and {bid}. Which case has the larger liver volume?',
           'expected':aid if volume_by_case[a]>volume_by_case[b] else bid,'unit':'case_id','tolerance':0}
        # Both start uncached: both must be obtained through real HTTP calls.
        await execute(t,'multi')
    dump('smoke_complete.json', {'completed':len(rows),'expected':len(tasks)*3+len(ids)})


if __name__ == '__main__':
    asyncio.run(main())
