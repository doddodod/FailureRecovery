"""Download an HDF5 manifest atomically from the Hugging Face mirror."""
import argparse, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=8); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
base='https://hf-mirror.com/datasets/nvidia/LIBERO-Cosmos-Policy/resolve/main/'
def download(line):
    rel=line.strip()
    if not rel: return
    target=a.output/Path(rel).name
    if target.exists() and target.stat().st_size>100000: return
    tmp=target.with_suffix(target.suffix+'.part')
    for attempt in range(20):
        result=subprocess.run(['curl','-sS','-L','--fail','--connect-timeout','20','--max-time','180',
                               '--retry','3','-o',str(tmp),base+rel])
        if result.returncode==0 and tmp.exists() and tmp.stat().st_size>100000:
            tmp.replace(target); return
        tmp.unlink(missing_ok=True); time.sleep(min(30,attempt+1))
    raise RuntimeError(f'failed after retries: {rel}')
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    list(pool.map(download,a.manifest.read_text().splitlines()))
