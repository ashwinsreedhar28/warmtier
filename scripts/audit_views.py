import sys, re, json, collections, os
import pyarrow.parquet as pq
sys.path.insert(0,'.')
from warmtier.sweagent import process_run, FILE_HDR, VIEW_CMDS, _command
def kinds(traj):
    out=[]; last=""; first_user=True
    for m in traj:
        t=m.get("text") or m.get("system_prompt") or ""
        if m["role"]=="ai": last=_command(t); continue
        if m["role"]=="system": continue
        if first_user: first_user=False; continue
        w=last.split()[0] if last else ""
        if FILE_HDR.search(t[:400]) and w in VIEW_CMDS:
            out.append(w if w!="edit" else ("edit_ok" if "File updated" in t else "edit_failed"))
    return out
src,dst=sys.argv[1],sys.argv[2]
agg=collections.Counter(); pf=pq.ParquetFile(src)
for g in range(pf.metadata.num_row_groups):
    for i,row in enumerate(pf.read_row_group(g, columns=["instance_id","trajectory","model_name","target","exit_status"]).to_pylist()):
        chunks, views, run = process_run(row, i)
        ks=kinds(row["trajectory"]); assert len(ks)==len(views), (len(ks),len(views))
        n=run[6]; agg['prefill']+=run[11]; agg['runs']+=1
        for k,v in zip(ks,views):
            agg[f'n|{k}']+=1; agg[f'cc|{k}']+=v[5]*v[9]
            if v[10] is not None: agg[f'nf|{k}']+=1; agg[f'sf|{k}']+=v[5]*(n-v[10])
            if v[11] is not None: agg[f'nw|{k}']+=1; agg[f'sw|{k}']+=v[5]*(n-v[11])
json.dump(agg, open(dst,'w'))
print(os.path.basename(src), agg['runs'])
