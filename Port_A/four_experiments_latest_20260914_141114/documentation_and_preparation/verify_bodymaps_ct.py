import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('E:/DeepTumour_agent_eval/live_portb_v1')
REV = 'a7ded8ce201467fb8031d0c6d6413caed6100d6d'
cid = 'BDMAP_00000065'
dest = ROOT / 'bodymaps_verified' / cid
dest.mkdir(parents=True, exist_ok=True)
api = f'https://huggingface.co/api/datasets/BodyMaps/AbdomenAtlas1.0Mini/tree/{REV}/{cid}'
entry = next(e for e in json.load(urllib.request.urlopen(api, timeout=30)) if e['path'].endswith('/ct.nii.gz'))
url = f'https://huggingface.co/datasets/BodyMaps/AbdomenAtlas1.0Mini/resolve/{REV}/{cid}/ct.nii.gz?download=true'
ct_path = dest / 'ct.nii.gz'
if not ct_path.exists():
    with urllib.request.urlopen(url, timeout=60) as src, (dest/'ct.part').open('wb') as out:
        shutil.copyfileobj(src,out,1024*1024)
    (dest/'ct.part').rename(ct_path)
digest = hashlib.file_digest(ct_path.open('rb'),'sha256').hexdigest()
assert digest == entry['lfs']['oid']
ct = nib.load(ct_path)
image = np.asanyarray(ct.dataobj)
results = {}
for name in ['liver','liver_lesion']:
    p = ROOT/'source_cases'/cid/f'{name}.nii.gz'
    mask = nib.load(p)
    data = np.asanyarray(mask.dataobj)>0
    results[name] = {'shape':list(mask.shape),'same_shape':mask.shape==ct.shape,
                     'same_affine':bool(np.allclose(mask.affine,ct.affine,atol=1e-5)),
                     'foreground_voxels':int(data.sum())}
    if name=='liver':
        liver=data
    else:
        lesion=data
assert all(r['same_shape'] and r['same_affine'] for r in results.values())
zs=np.where(liver.any(axis=(0,1)))[0]
slices=[int(np.quantile(zs,q)) for q in [.25,.5,.75]]
if lesion.any():
    slices[1]=int(np.argmax(lesion.sum(axis=(0,1))))
fig,axes=plt.subplots(1,3,figsize=(12,5))
for ax,z in zip(axes,slices):
    ax.imshow(image[:,:,z].T,cmap='gray',origin='lower',vmin=-150,vmax=250)
    ax.contour(liver[:,:,z].T,levels=[.5],colors=['lime'],linewidths=.8)
    if lesion[:,:,z].any():
        ax.contour(lesion[:,:,z].T,levels=[.5],colors=['red'],linewidths=1)
    ax.set_title(f'{cid} z={z}')
    ax.axis('off')
fig.tight_layout()
fig.savefig(dest/'overlay.png',dpi=130)
qa=json.loads((ROOT.parent/'source/test_qa.complete.json').read_text(encoding='utf-8'))['questions']
questions=[{k:q[k] for k in ['qid','question','answer','question_subtype']} for q in qa if q['image_id']==cid and q['question_subtype'] in {'liver lesion existence','lesion counting','organ volume measurement','lesion volume measurement','largest lesion diameter'} and q['organ']=='liver']
result={'case':cid,'ct_url':url,'ct_sha256':digest,'download_bytes':ct_path.stat().st_size,
        'ct_shape':list(ct.shape),'masks':results,'original_questions':questions,
        'visual_review':'pending','scope':'single-case validation, not whole-source approval'}
(dest/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
