"""Check published inputs, import isolation and source privacy without training."""
from __future__ import annotations

import argparse
import ast
import getpass
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

EXPECTED = {'ingolstadt_custom': (853,1035,306,1426),
            'montereau_fault_yonne': (4302,1500,283,8005),
            'coulommiers': (4483,1811,489,9083)}
METHODS = ('greedy','ga','iql','ippo','mappo','gnac','dreamer','mamba','mbpo')


def source_audit(root=ROOT):
    owner=getpass.getuser()
    private_root = re.compile(r'(?<![\w])/(?:' + '|'.join(
        ('home','Users','mnt','media','scratch','workspace','tmp','root','opt')) + r')/[\w.\-]+')
    windows_root = re.compile(r'\b[A-Za-z]:[\\/]')
    email = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
    forbidden_dirs={'.aws','.agents','.codex','.omx','__pycache__','.pytest_cache'}
    errors=[]
    python_files=0
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if '.git' in relative.parts:
            continue
        if path.is_symlink():
            errors.append((relative.as_posix(),'symlink'))
        if path.is_dir():
            if path.name in forbidden_dirs:
                errors.append((relative.as_posix(),'excluded runtime directory'))
            continue
        if path.suffix in ('.pyc','.pyo','.pt','.npz','.log'):
            errors.append((relative.as_posix(),'runtime artifact'))
        try:
            text=path.read_text()
        except (UnicodeDecodeError,OSError):
            errors.append((relative.as_posix(),'unexpected binary file'))
            continue
        if path.suffix=='.py':
            ast.parse(text,filename=relative.as_posix())
            python_files+=1
        if private_root.search(text) or windows_root.search(text) or email.search(text):
            errors.append((relative.as_posix(),'private path or email pattern'))
        if len(owner)>=6 and (owner in text or owner in relative.as_posix()):
            errors.append((relative.as_posix(),'local account identifier'))
    if errors:
        raise ValueError('release audit failed: '+json.dumps(errors))
    return dict(python_files=python_files,private_path_or_identity_matches=0,runtime_artifacts=0)


def validate_inputs():
    import urban_dreamer
    from tools.run_baseline import METHODS as available
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    if not Path(urban_dreamer.__file__).resolve().is_relative_to(ROOT):
        raise ValueError('package was imported from outside the release')
    if tuple(available)!=METHODS:
        raise ValueError('baseline set differs from the nine paper methods')
    expected_hashes=json.loads((ROOT/'configs/input_hashes.json').read_text())
    results={}
    for city,expected in EXPECTED.items():
        cfg=load_config(ROOT/'configs'/f'{city}.toml')
        if (cfg.run.online_training_episodes,cfg.run.evaluation_episodes)!=(1000,3):
            raise ValueError('paper episode budgets differ')
        for filename,digest in expected_hashes[city].items():
            with (cfg.inputs.network_dir/filename).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
                    raise ValueError('input hash mismatch: '+city+'/'+filename)
        prepared=validate_prepared_inputs(cfg)
        topology=load_traffic_topology(prepared)
        pairs=set()
        for _,element in ET.iterparse(prepared.network_xml,events=('end',)):
            if element.tag=='connection':
                source,target=element.get('from'),element.get('to')
                if source in topology.network.edge_to_index and target in topology.network.edge_to_index:
                    pairs.add((source,target))
            element.clear()
        actual=(topology.network.edge_count,len(topology.vehicles),
            len({(v.origin_index,v.destination_index) for v in topology.vehicles}),
            len(pairs))
        if actual!=expected:
            raise ValueError('manuscript network statistics mismatch for '+city+': '+str(actual))
        results[city]=dict(road_segments=actual[0],expected_vehicles=actual[1],active_ods=actual[2],
            legal_movements=actual[3],
            passenger_legal_movements=sum(len(targets) for targets in topology.network.adjacency_list),
            input_hashes_verified=True)
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only',action='store_true')
    args=parser.parse_args(argv)
    result=dict(status='passed',source=source_audit())
    if not args.source_only:
        result['cities']=validate_inputs()
        result['methods']=list(METHODS)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    main()
