"""Freeze and prepare the first formal live-Port-B DeepTumorVQA cohort."""
import concurrent.futures
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path('E:/DeepTumour_agent_eval/live_portb_formal_v1')
SOURCE = ROOT.parent/'source'
ROOT.mkdir(exist_ok=True)
QA_REV='a8b9db5133fda02567476dcb49498ec51f6aed37'
MASK_REV='122e04fc188949ed60a74f3bc6b4fa74e06a6152'
CT_REV='a7ded8ce201467fb8031d0c6d6413caed6100d6d'
PILOT={'BDMAP_00003877','BDMAP_00004228','BDMAP_00001998','BDMAP_00003420','BDMAP_00009168','BDMAP_00006864'}
SCOPE={'liver lesion existence','lesion counting','organ volume measurement','lesion volume measurement','largest lesion diameter'}
TOLERANCE={
 'organ volume measurement': {'absolute_floor':1.0,'relative':0.02,'unit':'cm3'},
 'lesion volume measurement': {'absolute_floor':0.1,'relative':0.05,'unit':'cm3'},
 'largest lesion diameter': {'absolute_floor':0.1,'relative':0.05,'unit':'cm'},
 'lesion counting': {'absolute':0,'unit':'count'},
 'liver lesion existence': {'exact':True,'unit':'bool'},
}

qa=json.loads((SOURCE/'test_qa.complete.json').read_text(encoding='utf-8'))['questions']
questions=[q for q in qa if q['organ']=='liver' and q['question_subtype'] in SCOPE and q['image_id'] not in PILOT
           and int(q['image_id'].split('_')[-1])<=1624 and 'free of lesions or' not in q['question'].lower()]
questions.sort(key=lambda q:q['qid'])
ids=sorted({q['image_id'] for q in questions})
manifest={'protocol':'live_portb_formal_v1','frozen_before_download':True,'questions':len(questions),'cases':len(ids),
          'by_subtype':dict(Counter(q['question_subtype'] for q in questions)),'case_ids':ids,
          'cohort_rule':'all eligible original questions for case IDs 1-1624; contiguous range covered by the first seven verified mask shards',
          'cohort_revision_reason':'shards 8-9 were unavailable after repeated network failures before any formal model run; selection was not based on model outputs',
          'source_revisions':{'qa':QA_REV,'mask':MASK_REV,'ct':CT_REV},'answer_agreement_tolerances':TOLERANCE,
          'port_b_numerical_validation':{'volume_absolute_cm3':0.01,'count':'exact','existence':'exact',
             'diameter_note':'independent algorithm validation is outside this agent experiment; compare to original answer using the the frozen agreement tolerance'},
          'scope':'same-grid BodyMaps CT and AbdomenAtlas3.0Mini reference masks; actual Port B HTTP computation; no automatic segmentation'}
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'original_questions_smoke.json').write_text(json.dumps(questions,ensure_ascii=False,indent=2),encoding='utf-8')

def retrieve(url,dest,size=None,sha=None):
    part=dest.with_suffix(dest.suffix+'.part')
    for attempt in range(5):
        offset=part.stat().st_size if part.exists() else 0
        headers={'Range':f'bytes={offset}-'} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as src:
                if offset and src.status!=206:
                    part.unlink(); offset=0; return retrieve(url,dest,size,sha)
                with part.open('ab' if offset else 'wb') as out:
                    shutil.copyfileobj(src,out,1024*1024)
            if size is not None and part.stat().st_size!=size: raise IOError('size mismatch')
            if sha is not None and hashlib.file_digest(part.open('rb'),'sha256').hexdigest()!=sha: raise IOError('hash mismatch')
            part.replace(dest); return
        except Exception:
            if attempt==4: raise
            time.sleep(2*(attempt+1))

needed_shards=sorted({(int(cid.split('_')[-1])-1)//232 for cid in ids})
archives=[]
for shard in needed_shards:
    start=shard*232+1; end=(shard+1)*232
    dest=ROOT/f'AbdomenAtlas3_masks_BDMAP_BDMAP_{start:08d}_BDMAP_{end:08d}.tar.gz'
    if not dest.exists():
        mask_entries=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini/tree/{MASK_REV}/mask_only',timeout=30))
        entry=mask_entries[shard]
        url=f'https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini/resolve/{MASK_REV}/{entry["path"]}'
        retrieve(url,dest,entry['size'],entry['lfs']['oid'])
    archives.append(dest)
    print('MASK_SHARD',shard+1,'READY',flush=True)

needed=set(ids)
for archive_path in archives:
    with tarfile.open(archive_path) as archive:
        for member in archive:
            parts=Path(member.name).parts
            if member.isfile() and len(parts)==3 and parts[0] in needed and parts[2] in {'liver.nii.gz','liver_lesion.nii.gz'}:
                dest=ROOT/'source_cases'/parts[0]/parts[2]; dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,dest.open('wb') as out: shutil.copyfileobj(src,out)

def fetch_ct(cid):
    dest=ROOT/'source_cases'/cid/'ct.nii.gz'
    api=f'https://huggingface.co/api/datasets/BodyMaps/AbdomenAtlas1.0Mini/tree/{CT_REV}/{cid}'
    entry=next(e for e in json.load(urllib.request.urlopen(api,timeout=30)) if e['path'].endswith('/ct.nii.gz'))
    url=f'https://huggingface.co/datasets/BodyMaps/AbdomenAtlas1.0Mini/resolve/{CT_REV}/{cid}/ct.nii.gz?download=true'
    retrieve(url,dest,entry['size'],entry['lfs']['oid'])
    print('CT',cid,'READY',flush=True)
    return cid,entry['lfs']['oid']

hashes=dict(concurrent.futures.ThreadPoolExecutor(max_workers=6).map(fetch_ct,ids))
records=[]
for cid in ids:
    base=ROOT/'source_cases'/cid; ct=nib.load(base/'ct.nii.gz'); masks={}; ok=True
    for name in ['liver','liver_lesion']:
        m=nib.load(base/f'{name}.nii.gz'); same=m.shape==ct.shape and np.allclose(m.affine,ct.affine,atol=1e-5); ok &= same
        data=np.asanyarray(m.dataobj)
        masks[name]={'shape':list(m.shape),'same_grid':bool(same),'foreground':int(np.count_nonzero(data)),
          'volume_cm3':float(np.count_nonzero(data)*abs(np.linalg.det(m.affine[:3,:3]))/1000)}
    records.append({'case':cid,'ct_shape':list(ct.shape),'ct_sha256':hashes[cid],'masks':masks,'geometry_passed':bool(ok)})
    print('GEOMETRY',cid,ok,flush=True)
(ROOT/'geometry_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
passed=[r for r in records if r['geometry_passed']]
(ROOT/'ready.json').write_text(json.dumps({'cases':len(passed),'questions':sum(q['image_id'] in {r['case'] for r in passed} for q in questions)},indent=2),encoding='utf-8')
failed=[r['case'] for r in records if not r['geometry_passed']]
eligible={r['case'] for r in passed}
ready_questions=[q for q in questions if q['image_id'] in eligible]
(ROOT/'original_questions_smoke.json').write_text(json.dumps(ready_questions,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'data_exclusions.json').write_text(json.dumps({'geometry_failed_cases':failed,'excluded_before_model_run':True},indent=2),encoding='utf-8')
print('FORMAL_DATA_READY',len(passed),len(ready_questions),'EXCLUDED',failed,flush=True)
