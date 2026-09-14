"""Rerun the original 27 volume questions over the real Port B HTTP API."""
import os
os.environ['LIVE_EVAL_ROOT'] = 'E:/DeepTumour_agent_eval/volume_retest'
os.environ['LIVE_EVAL_DATA_ROOT'] = 'E:/DeepTumour_agent_eval/live_portb_formal_v1'
import run_live_portb_smoke as base
import asyncio
import json
import time
import re

async def main():
    base.ROOT.mkdir(exist_ok=True)
    tasks = [t for t in json.loads((base.DATA_ROOT/'smoke_tasks.json').read_text(encoding='utf-8')) if t['subtype']=='lesion volume measurement']
    skills = await base.server.port_b_list_skills(force_refresh=True)
    sem = asyncio.Semaphore(2)
    rows=[]
    async def run(t):
        async with sem:
            s=base.server.create_empty_session('volume_retest_'+t['id'])
            s.update({'case_ids':t['cases'],'_active_case_ids':t['cases'],'_active_volumes':[c+'.nii.gz' for c in t['cases']], 'available_skills':skills['skills'],'available_tools':base.server.inject_target_case_id(skills['tools'],t['cases']),'max_new_tokens':700,'segmentation_status':'completed','original_question':t['query']})
            for cid in t['cases']:
                s['tool_store'][cid]={'segmentation':{'status':'reference_masks','case_id':cid}}
            s['current_user_query']=t['query']+'\nAnswer concisely. End with one JSON object: {"value": <number, true, false, null if unavailable, or case ID for comparison>, "unit": "cm3"}. Do not guess missing evidence.'
            started=time.monotonic()
            try:
                answer=await asyncio.wait_for(base.server.run_agent_loop(base.Sink(),s,max_rounds=4),180)
                value=json.loads(re.findall(r'\{[^{}]*"value"\s*:[^{}]*\}',answer)[-1])['value']
                totals=[c.get('result',{}).get('total_tumor_volume_cm3') for c in s['skill_call_history'] if isinstance(c.get('result'),dict)]
                total=next((v for v in totals if isinstance(v,(int,float))),None)
                row={'id':t['id'],'question':t['query'],'expected':t['expected'],'answer':answer,'value':value,'score':base.score(answer,t['expected'],'cm3',t['tolerance']),'tool_total':total,'complete_total_match':total is not None and isinstance(value,(int,float)) and abs(total-value)<1e-6}
            except Exception as exc:
                row={'id':t['id'],'error':str(exc)}
            row.update(calls=list(s['skill_call_history']),seconds=time.monotonic()-started)
            rows.append(row)
            (base.ROOT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            print(t['id'],row.get('score'),row.get('complete_total_match'),flush=True)
    await asyncio.gather(*(run(t) for t in tasks))
    summary={'n':len(rows),'numeric':sum(isinstance(r.get('value'),(int,float)) for r in rows),'vqa_pass':sum(r.get('score',{}).get('answer_correct',False) for r in rows),'complete_total_match':sum(r.get('complete_total_match',False) for r in rows),'exceptions':sum('error' in r for r in rows)}
    (base.ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)

if __name__=='__main__':
    asyncio.run(main())
