import json
import re
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path('E:/DeepTumour_agent_eval/live_portb_formal_v1')
OUT = Path('E:/DeepTumour_agent_eval/volume_audit')
OUT.mkdir(exist_ok=True)
tasks = [t for t in json.loads((ROOT/'smoke_tasks.json').read_text(encoding='utf-8')) if t['subtype'] == 'lesion volume measurement']
results = {r['id']: r for r in map(json.loads, (ROOT/'smoke_results.jsonl').read_text(encoding='utf-8').splitlines())}
cache = {}
rows = []
for t in tasks:
    cid = t['source_case']
    if cid not in cache:
        source = nib.load(ROOT/'source_cases'/cid/'liver_lesion.nii.gz')
        sm = np.asarray(source.dataobj) > 0
        sv = float(sm.sum()*abs(np.linalg.det(source.affine[:3,:3]))/1000)
        parts = {}
        union = np.zeros(sm.shape, dtype=bool)
        for p in sorted((ROOT/'working_cases'/('LIVE_'+cid)/'masks').glob('tumor_*.nii.gz')):
            n = nib.load(p)
            m = np.asarray(n.dataobj)>0
            assert m.shape == sm.shape and np.allclose(n.affine, source.affine)
            union |= m
            parts[p.name.removesuffix('.nii.gz')] = float(m.sum()*abs(np.linalg.det(n.affine[:3,:3]))/1000)
        cache[cid] = dict(source_volume_ml=sv, parts=parts, split_total_ml=sum(parts.values()), union_matches_source=bool(np.array_equal(union,sm)), overlap_voxels=int(sum(round(v*1000/abs(np.linalg.det(source.affine[:3,:3]))) for v in parts.values())-union.sum()))
    r = results['cold_'+t['id']]
    matches = re.findall(r'\{[^{}]*"value"\s*:[^{}]*\}',r.get('answer',''))
    value = json.loads(matches[-1])['value'] if matches else None
    calls = [c for c in r['calls'] if isinstance(c.get('result'),dict) and 'tumor_results' in c['result']]
    tool = calls[-1]['result']['tumor_results'] if calls else {}
    tool_total = sum(v['volume_cm3'] for v in tool.values()) if tool else None
    rounded_match = all(abs(v['volume_cm3']-round(cache[cid]['parts'][k],2))<1e-8 for k,v in tool.items()) if tool else None
    row = dict(id=t['id'],case=cid,question=t['query'],vqa_ml=t['expected'],tolerance_ml=t['tolerance'],old_pass=r['score']['answer_correct'],answer_ml=value,tool_total_rounded_ml=tool_total,tool_count=len(tool),tool_each_matches_mask_rounded=rounded_match,**cache[cid])
    row['source_vqa_relative_error_pct']=100*abs(row['source_volume_ml']-t['expected'])/t['expected']
    row['answer_source_error_ml']=abs(value-row['source_volume_ml']) if isinstance(value,(float,int)) else None
    row['answer']=r.get('answer')
    row['calls']=r['calls']
    rows.append(row)
(OUT/'volume_27_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 原27道病灶体积题逐题复核','', '体积单位均为mL，与cm³等价。原记录保持不变；参考体积由上游掩膜独立计算。','', '| 题号 | 病例 | VQA | 掩膜实算 | 完整工具求和 | 原回答 | 原评分 |','|---|---|---:|---:|---:|---:|---|']
for r in rows:
    f=lambda v: 'null' if v is None else f'{v:.5f}'
    lines.append(f"|{r['id']}|{r['case']}|{f(r['vqa_ml'])}|{f(r['source_volume_ml'])}|{f(r['tool_total_rounded_ml'])}|{f(r['answer_ml'])}|{'通过' if r['old_pass'] else '未通过'}|")
for r in rows:
    lines += ['',f"## {r['id']} · {r['case']}",r['question'],'',f"掩膜拆分无损：{r['union_matches_source']}；重叠体素：{r['overlap_voxels']}；工具逐病灶数值符合掩膜四舍五入结果：{r['tool_each_matches_mask_rounded']}；掩膜与VQA相对差异：{r['source_vqa_relative_error_pct']:.4f}%。",'',r['answer']]
(OUT/'volume_27_audit.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k not in ('calls','answer','parts')} for r in rows],ensure_ascii=False,indent=2))
