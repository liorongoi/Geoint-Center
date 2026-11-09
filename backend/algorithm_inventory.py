"""Static inventory for a declared offline algorithm package."""
import ast
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath

FORMATS = {
    'source': {'.py'}, 'config': {'.json'}, 'weights': {'.safetensors', '.pth', '.pt', '.onnx'},
    'license': {'.txt', '.md'}, 'dependencies': {'.txt', '.json'},
}
SOURCE_INSPECTION_LIMIT = 2 * 1024**2


def source_interface(raw):
    """Describe declarations only. Defaults and annotations are never evaluated."""
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError('Python source could not be parsed for the local runtime') from exc
    functions, imports = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add('.' * node.level + (node.module or ''))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(functions) < 500:
            args = node.args
            parameters = [arg.arg for arg in [*args.posonlyargs, *args.args]]
            if args.vararg:
                parameters.append('*' + args.vararg.arg)
            parameters.extend(arg.arg for arg in args.kwonlyargs)
            if args.kwarg:
                parameters.append('**' + args.kwarg.arg)
            functions.append({'name':node.name,'parameters':parameters,'line':node.lineno})
    return {'functions':functions,'imports':sorted(imports)[:500]}


def bounded_json(path, limit, *, reject_duplicates=False):
    with path.open('rb') as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError('JSON document exceeds the inspection limit')
    def unique(pairs):
        result=dict(pairs)
        if len(result)!=len(pairs):raise ValueError('JSON document contains duplicate fields')
        return result
    try:
        document=json.loads(content,object_pairs_hook=unique if reject_duplicates else None)
    except RecursionError as exc:
        raise ValueError('JSON document nesting exceeds the inspection limit') from exc
    return document, hashlib.sha256(content).hexdigest()


def tensor_inventory(stream, size):
    header_size = int.from_bytes(stream.read(8), 'little')
    if not 2 <= header_size <= min(1024**2, size-8):
        raise ValueError('Invalid or oversized SafeTensors header')
    def unique_object(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError('SafeTensors header contains duplicate keys')
        return value

    header = json.loads(stream.read(header_size), object_pairs_hook=unique_object)
    if not isinstance(header, dict) or len(header) > 10001:
        raise ValueError('Invalid SafeTensors tensor inventory')
    result, spans = [], []
    dtype_size = {'BOOL':1,'U8':1,'I8':1,'U16':2,'I16':2,'F16':2,'BF16':2,
                  'U32':4,'I32':4,'F32':4,'U64':8,'I64':8,'F64':8}
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(tensor, dict) or set(tensor) != {'dtype','shape','data_offsets'}:
            raise ValueError('Invalid SafeTensors tensor declaration')
        shape, offsets = tensor['shape'], tensor['data_offsets']
        width = dtype_size.get(tensor['dtype']) if isinstance(tensor['dtype'], str) else None
        if (width is None or not isinstance(shape, list) or len(shape)>16
                or any(type(n) is not int or not 0<=n<=2**32 for n in shape)
                or not isinstance(offsets,list) or len(offsets)!=2
                or any(type(n) is not int for n in offsets)
                or not 0<=offsets[0]<=offsets[1]<=size-8-header_size):
            raise ValueError('Unsupported SafeTensors dtype, shape or byte range')
        from math import prod
        if prod(shape)*width != offsets[1]-offsets[0]:
            raise ValueError('SafeTensors shape does not match its byte range')
        spans.append(offsets)
        result.append({'name':name,'shape':shape,'dtype':tensor['dtype']})
    position = 0
    for start,end in sorted(spans):
        if start!=position:
            raise ValueError('SafeTensors byte ranges overlap or leave a gap')
        position=end
    if position!=size-8-header_size or not result:
        raise ValueError('SafeTensors contains unaccounted data or no tensors')
    return {'tensor_count':len(result), 'tensor_sample':result[:25]}


def inspect_package(services, path, task):
    manifest = services.source(path,{'.json'},256*1024)
    document, checksum = bounded_json(manifest,256*1024)
    if (not isinstance(document,dict) or set(document)!={'schema','name','task','artifacts'}
            or document['schema']!='geo-intelligence-algorithm-inventory-v1' or document['task']!=task
            or not isinstance(document['name'],str) or not 1<=len(document['name'].strip())<=120
            or not isinstance(document['artifacts'],list) or not 1<=len(document['artifacts'])<=32):
        raise ValueError('Use a Geo Intelligence algorithm inventory with a matching task and 1–32 explicit artifacts')
    inventory, seen, total = [],set(),0
    interfaces, source_budget = [], 8 * 1024**2
    for artifact in document['artifacts']:
        if (not isinstance(artifact,dict) or set(artifact)!={'role','path'}
                or not isinstance(artifact['role'],str) or artifact['role'] not in FORMATS
                or not isinstance(artifact['path'],str)):
            raise ValueError('Each artifact needs a supported role and relative path')
        relative = PurePosixPath(artifact['path'])
        if (relative.is_absolute() or '..' in relative.parts or '\\' in str(relative)
                or ':' in str(relative) or len(str(relative))>512 or relative in seen):
            raise ValueError('Use unique relative artifact paths inside the package')
        role = artifact['role']
        source = services.source(manifest.parent / Path(*relative.parts),FORMATS[role],2**31)
        if not source.is_relative_to(manifest.parent):
            raise ValueError('Artifact is outside the package')
        size = source.stat().st_size
        total += size
        if total>4*1024**3:
            raise ValueError('Algorithm inventory exceeds 4 GiB')
        record = {'role':role,'path':relative.as_posix(),'bytes':size}
        with source.open('rb') as stream:
            before = source.stat()
            if source.suffix.lower()=='.safetensors':
                record.update(tensor_inventory(stream,size))
                stream.seek(0)
            if role=='config':
                config, _ = bounded_json(source,2*1024**2)
                if not isinstance(config,dict):
                    raise ValueError('Use an object for algorithm configuration')
                record['config_keys'] = sorted(config)[:100]
            if role=='source' and size<=min(SOURCE_INSPECTION_LIMIT,source_budget):
                raw=stream.read(size+1)
                if len(raw)!=size:
                    raise ValueError('Artifact changed during inspection')
                source_budget-=size
                record['sha256']=hashlib.sha256(raw).hexdigest()
                try:
                    interfaces.append({'path':record['path'],**source_interface(raw)})
                except ValueError as exc:
                    interfaces.append({'path':record['path'],'inspection_error':str(exc)})
            else:
                record['sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
                if role=='source':
                    interfaces.append({'path':record['path'],'inspection_error':'Source exceeds the declaration inspection budget'})
            after = source.stat()
            if (before.st_mtime_ns,before.st_size,before.st_ino)!=(after.st_mtime_ns,after.st_size,after.st_ino):
                raise ValueError('Artifact changed during inspection')
        inventory.append(record)
        seen.add(relative)
    if bounded_json(manifest,256*1024)[1]!=checksum:
        raise ValueError('Algorithm manifest changed during inspection')
    return {'name':document['name'].strip(),'source':str(manifest),'source_sha256':checksum,
            'functions':[],'imports':[],'artifacts':inventory,'source_interfaces':interfaces,'inventory_schema':document['schema']}
