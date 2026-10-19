"""Event-driven safe checkpoint stop, not a scientific replay or source mutation."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import struct
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def run():
    mode=sys.argv[1];pid=int(sys.argv[2]);stem=PREFIX+mode;target=stem+'_preparation.npz';start=time.monotonic();cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'rich_photon_collective_20261015_coordinate.py' not in cmdline or mode not in cmdline:raise RuntimeError('Child PID binding does not match coordinate consumer')
    libc=ctypes.CDLL(None,use_errno=True);notify=libc.inotify_init1(os.O_CLOEXEC)
    if notify<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    descriptor=libc.inotify_add_watch(notify,os.fsencode(ROOT),0x00000008)
    if descriptor<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
    pidfd=os.pidfd_open(pid);trigger=None
    if (ROOT/target).exists() and zipfile.is_zipfile(ROOT/target):trigger='already_completed_preparation'
    while trigger is None:
        ready,_,_=select.select([notify,pidfd],[],[])
        if pidfd in ready:trigger='child_exited_before_preparation';break
        events=os.read(notify,65536);offset=0
        while offset<len(events):
            _,mask,_,length=struct.unpack_from('iIII',events,offset);offset+=16;name=events[offset:offset+length].split(b'\0',1)[0].decode();offset+=length
            if mask&0x00000008 and name==target:trigger='preparation_closed_before_evolution';break
    record=dict(mode=mode,PID=pid,command=cmdline,trigger=trigger,elapsed_seconds=time.monotonic()-start,reason='Known dt1/K28 propagation has large physical endpoint residuals in the larger-ring spectrum. Preserve complete coordinate modes/source/preparation and prevent auto-entry into the same inaccurate trajectory. This is interrupted, NOT natural scientific completion.',source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if trigger!='child_exited_before_preparation':
        os.kill(pid,signal.SIGSTOP);record['signals']=['SIGSTOP','SIGINT','SIGCONT'];record['preparation_SHA256']=hashlib.sha256((ROOT/target).read_bytes()).hexdigest();os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT)
    (ROOT/(stem+'_control_receipt.json')).write_text(json.dumps(record,indent=2)+'\n');os.close(pidfd);os.close(notify);print(json.dumps(record),flush=True)

if __name__=='__main__':run()
