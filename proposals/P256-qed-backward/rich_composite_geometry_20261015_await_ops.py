"""One finite inotify wait for an EXISTING producer: no polling or computation.
Returns when its complete operator descriptor closes, or its exit receipt does.
Does not start/stop a producer or launch any scientific consumer.
"""
import ctypes,json,os,pickletools,struct,sys
from pathlib import Path


def complete_descriptor(path):
    if not path.is_file():return False
    try:
        data=path.read_bytes();last=None
        for opcode,argument,position in pickletools.genops(data):last=(opcode.name,position)
        return last==('STOP',len(data)-1)
    except (ValueError,EOFError,OSError):return False


def main():
    prefix=Path(sys.argv[1]).resolve();directory=prefix.parent
    descriptor=Path(str(prefix)+'_operators.pkl');terminal=Path(str(prefix)+'_exit.json')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.inotify_init1.argtypes=[ctypes.c_int];libc.inotify_init1.restype=ctypes.c_int
    libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];libc.inotify_add_watch.restype=ctypes.c_int
    fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if libc.inotify_add_watch(fd,os.fsencode(directory),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while True:
            if complete_descriptor(descriptor):
                print(json.dumps({'event':'EXISTING_NATIVE_OPERATORS_USABLE_BEFORE_POLES','descriptor':str(descriptor),'meaning':'Readiness only; no scientific result, no new constructor or consumer launch.'}),flush=True)
                return
            if terminal.is_file():raise RuntimeError('Existing producer exited without a complete owning operator descriptor: '+str(terminal))
            events=os.read(fd,65536);offset=0;relevant=False
            while offset<len(events):
                _,_,_,length=struct.unpack_from('iIII',events,offset);offset+=16
                name=events[offset:offset+length].split(b'\0',1)[0];offset+=length
                if name in (os.fsencode(descriptor.name),os.fsencode(terminal.name)):relevant=True
            if not relevant:
                # Wait for the next kernel event; no repeated file/stat checks.
                while not relevant:
                    events=os.read(fd,65536);offset=0
                    while offset<len(events):
                        _,_,_,length=struct.unpack_from('iIII',events,offset);offset+=16
                        name=events[offset:offset+length].split(b'\0',1)[0];offset+=length
                        if name in (os.fsencode(descriptor.name),os.fsencode(terminal.name)):relevant=True
    finally:os.close(fd)


if __name__=='__main__':main()
