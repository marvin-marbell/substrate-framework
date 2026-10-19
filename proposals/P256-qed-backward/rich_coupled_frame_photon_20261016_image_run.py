"""Independent exact unbounded-H images on the produced fresh packets."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import rich_coupled_frame_photon_20261016_model as m
from rich_coupled_frame_photon_20261016_images import full_image

ROOT=Path(__file__).resolve().parent
P='rich_coupled_frame_photon_20261016_'


def main():
    start=time.time();result={}
    for number,name in [(4,'neutral'),(5,'addition'),(3,'removal')]:
        model=m.Model(number)
        with np.load(ROOT/(P+name+'_raw.npz')) as raw:
            if name=='neutral':states=[raw['neutral_input'],raw['neutral_states'][-1]]
            else:states=[raw[name+'_input'],raw[name+'_end'][:,0],raw[name+'_end'][:,1]]
            result[name]=[full_image(model,state) for state in states]
        del model
    result['seconds']=time.time()-start
    result['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/(P+'model.py'),ROOT/(P+'images.py'),ROOT/(P+'image_run.py')]}
    (ROOT/(P+'images_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
