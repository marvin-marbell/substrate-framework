"""Scoped existing direct Pyright protocol and positive Python AST receipts."""
import ast
import json
from pathlib import Path
import sys
from rich_review_wave12_20261015_diagnose import language_server
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    mode=sys.argv[1]
    files=sys.argv[2:]
    record=language_server(files)
    record['positive_AST']={name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for name in files}
    (ROOT/f'rich_frame_field_join_20261015_language_{mode}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'mode':mode,'symbol_files':len(record['symbols']),'server_exit':record['server_exit'],'AST_matches':{k:len(v) for k,v in record['positive_AST'].items()}}))
