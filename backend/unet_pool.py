"""Synchronous U-Net data parallelism over enrolled, leased worker tasks."""
import json
import os
from pathlib import Path
import time
from uuid import UUID, uuid5

import numpy as np

import distributed_execution as pool
from agent_services import file_hash
from distributed_queue import Queue
from neural_training_sync import (
    apply_round, gradient, initial_state, load_round, read_tensors, save_round, write_tensors,
)

RAM_GIB = 5
VRAM_GIB = 4
MAX_GRID = 384
MAX_IMAGES = 8
MAX_SAMPLE_BYTES = 64 * 1024**2


def select_workers(workers, count):
    eligible = [w for w in workers if w['ram_gib'] >= RAM_GIB
                and any(d['kind'] == 'cpu' or d['vram_gib'] >= VRAM_GIB for d in w['devices'])]
    eligible.sort(key=lambda w: not any(d['kind'] != 'cpu' and d['vram_gib'] >= VRAM_GIB for d in w['devices']))
    if len(eligible) < count:
        raise ValueError(f'Synchronized U-Net needs {count} distinct enrolled workers with at least {RAM_GIB} GiB budget each')
    return [w['id'] for w in eligible[:count]]


def _immutable_tensors(path, values):
    try:
        write_tensors(path, values)
    except FileExistsError:
        saved = _read_sample(path)
        if set(saved) != set(values) or any(not np.array_equal(value, saved[name]) for name, value in values.items()):
            raise ValueError('Previously committed U-Net input differs; existing evidence retained')


def _read_sample(path):
    from safetensors.numpy import load

    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_SAMPLE_BYTES + 1)
    if len(raw) > MAX_SAMPLE_BYTES:
        raise ValueError('U-Net sample exceeds its offline transfer budget')
    return load(raw)


def work_partition(run, spec, out, persist, device_kind):
    """Only numeric, partitioned fitting inputs are available to this child."""
    import torch

    torch.set_num_threads(2)
    if (set(spec) != {'weights', 'samples', 'channels', 'classes', 'round', 'rank'}
            or not 1 <= len(spec['samples']) <= MAX_IMAGES):
        raise ValueError('Invalid typed U-Net gradient task')
    schema = initial_state(spec['channels'], spec['classes'])
    weights = read_tensors(Path(spec['weights']), schema)
    counts = 0
    loss = 0.
    total = {name: np.zeros_like(value, dtype='float64') for name, value in weights.items()}
    for i, path in enumerate(spec['samples']):
        persist({'status': 'running', 'progress': 10+80*i/len(spec['samples']),
                 'stage': f'U-Net round {spec["round"]} · backpropagating sample {i+1}'})
        sample = _read_sample(path)
        if set(sample) != {'pixels', 'mask'} or max(sample['mask'].shape) > MAX_GRID:
            raise ValueError('Invalid U-Net fitting sample schema or grid')
        values, count, batch_loss, actual = gradient(weights, sample['pixels'], sample['mask'], spec['classes'])
        if actual.split(':', 1)[0] != device_kind:
            raise ValueError('Actual training device does not match the worker lease')
        for name, value in values.items():
            total[name] += value
        counts += count
        loss += batch_loss
    persist({'status': 'running', 'progress': 95, 'stage': 'Saving bounded numeric gradients'})
    write_tensors(out/'gradients.safetensors', {name: value.astype('float32') for name, value in total.items()})
    return {**run, 'status': 'complete', 'round': spec['round'], 'rank': spec['rank'],
            'fitting_pixels': counts, 'loss_sum': loss, 'actual_device': actual}


def _prepare(samples, classes, workers, folder):
    from model_workers import _balanced_samples

    if not 1 <= len(samples) <= MAX_IMAGES:
        raise ValueError(f'Synchronized U-Net supports up to {MAX_IMAGES} labeled images per candidate')
    bands = samples[0][0].bands
    if any(stack.bands != bands for stack, _, _ in samples):
        raise ValueError('Synchronized U-Net requires matching science-band schemas')
    parts = [_balanced_samples(stack.pixels, mask) for stack, mask, _ in samples]
    targets = np.concatenate([y for _, y in parts])
    if any(np.count_nonzero(targets == value) < 5 for value in range(len(classes))):
        raise ValueError('Every U-Net class needs at least five valid fitting pixels')
    features = np.concatenate([x for x, _ in parts])
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-6)
    count = len(workers)
    shards = [[] for _ in workers]
    sizes = [0] * count
    # Whole acquisitions avoid duplicate compute. When fewer images are
    # available, split the fitting pixels by contiguous rows and keep context
    # aligned to the U-Net's two pooling strides.
    for index, (stack, mask, excluded) in enumerate(samples):
        positions = np.flatnonzero(mask.ravel() >= 0)
        pixels = np.nan_to_num((stack.pixels-mean[:, None, None])/std[:, None, None]).astype('float32')
        pixels[:,excluded | ~stack.valid] = 0
        if len(samples) >= count:
            partitions = [(min(range(count),key=sizes.__getitem__),positions)]
        else:
            partitions = list(enumerate(np.array_split(positions,count)))
        for rank,selected in partitions:
            if not len(selected):
                continue
            partition = np.full(mask.shape, -1, dtype='int64')
            partition.ravel()[selected] = mask.ravel()[selected]
            height,width = mask.shape
            if len(partitions)>1 and height%4==0 and width%4==0:
                first,last = selected[[0,-1]]//width
                top,bottom = max(0,int(first)//4*4-32),min(height,(int(last)//4+1)*4+32)
            else:
                top,bottom = 0,height
            chip = pixels[:,top:bottom,:].copy()
            fit_mask = partition[top:bottom,:].copy()
            if chip.nbytes + fit_mask.nbytes > MAX_SAMPLE_BYTES - 4096:
                raise ValueError('U-Net sample exceeds its transfer budget; choose a smaller grid')
            path = folder/f'fit-{rank}-{index}.safetensors'
            _immutable_tensors(path, {'pixels':chip, 'mask':fit_mask})
            shards[rank].append(path)
            sizes[rank] += len(selected)
    if not all(sizes):
        raise ValueError('Every selected worker needs a nonempty fitting partition')
    return bands, mean, std, shards, sizes


def _verified_gradient(database, data, task, expected, schema):
    if task['status'] == 'awaiting_verification':
        task = Queue(database).verify_result(task['id'], pool.root(data)/'received', pool.root(data)/'verified')
    if task['status'] != 'complete' or task['worker_id'] != expected['worker']:
        raise ValueError('Gradient lease does not match the assigned U-Net partition')
    path = Path(task['verification']['directory'])
    receipt = json.loads((path/'verification.json').read_text())
    files = {entry['path']: entry for entry in receipt['files']}
    if set(files) != {'report.json', 'gradients.safetensors'}:
        raise ValueError('U-Net worker returned unexpected artifacts')
    for name, item in files.items():
        if (path/name).stat().st_size != item['bytes'] or file_hash(path/name) != item['sha256']:
            raise ValueError('A verified U-Net worker artifact changed')
    if files['report.json']['bytes'] > 8192:
        raise ValueError('U-Net gradient receipt exceeds its metadata budget')
    report = json.loads((path/'report.json').read_text())
    if (report.get('id') != task['payload']['run_id'] or report.get('requested_models') != ['unet']
            or report.get('status') != 'complete' or report.get('round') != expected['round']
            or report.get('rank') != expected['rank'] or report.get('fitting_pixels') != expected['pixels']
            or report.get('actual_device','').split(':',1)[0] != task['device_kind']):
        raise ValueError('Gradient receipt does not match the pinned U-Net task')
    return ((read_tensors(path/'gradients.safetensors', schema), report['fitting_pixels'], report['loss_sum']),
            {'task_id':task['id'], 'worker_id':task['worker_id'], 'attempt':task['attempt'],
             'device':report['actual_device'], 'leased_device':task['device'], 'fitting_pixels':report['fitting_pixels'],
             'gradient_sha256':files['gradients.safetensors']['sha256']})


def _round_tasks(database, data, run, workers, weights_file, shards, sizes, bands, classes, epoch, validation):
    tasks = []
    for rank, worker in enumerate(workers):
        identifier = str(uuid5(UUID(run['id']), f'unet:{epoch}:{rank}'))
        request = {'id':identifier, 'mission_id':run['mission_id'], 'job_id':run['job_id'],
                   'requested_models':['unet'], 'run_kind':'training'}
        spec = {'weights':str(weights_file), 'samples':[str(path) for path in shards[rank]],
                'channels':len(bands), 'classes':len(classes), 'round':epoch, 'rank':rank}
        manifest = pool.bundle(data, request, [], {}, [], {'unet_gradient':spec}, extra=[weights_file, *shards[rank]])
        task = Queue(database).enqueue('unet','training',
            {'run_id':identifier,'mission_id':run['mission_id'],'job_id':run['job_id'],
             'input_manifest':manifest,'training_version':run['id'],'validation_manifest':validation},
            f'{run["id"]}:unet:{epoch}:{rank}',ram_gib=RAM_GIB,vram_gib=VRAM_GIB,assigned_worker=worker)
        if task['payload']['input_manifest'] != manifest or task.get('assigned_worker') != worker:
            raise ValueError('Recovered U-Net task differs from its pinned partition')
        tasks.append(task['id'])
    return tasks


def execute(database, run, persist, images, access, labels, data, options, shutdown):
    """Recover from durable barrier records and re-enter the ordinary result path."""
    from job_runner import execute_job
    import mission_control

    queue = Queue(database)
    config = run['worker_pool_configuration']
    plan = run.get('distributed_plan')
    workers = run['unet_workers']
    folder = pool.root(data)/'unet'/run['id']
    start = time.monotonic()

    def control():
        if shutdown and shutdown.is_set():
            raise pool.CoordinatorRestart()
        if pool.desired(database, run)=='stop':
            raise mission_control.MissionStopped()

    def progress(value):
        control()
        while pool.desired(database,run)=='pause':
            persist({**value,'status':'paused','stage':'Paused at U-Net coordinator checkpoint'})
            time.sleep(.25)
            control()
        persist(value)

    try:
        control()
        if not plan:
            if not images or options['max_size'] > MAX_GRID or len(images) > MAX_IMAGES:
                raise ValueError(f'Synchronized U-Net supports up to {MAX_IMAGES} images at a {MAX_GRID} px grid')
            original = pool.bundle(data,run,images,access,labels,options)
            plan = {'phase':'unet_sync','original_bundle':original,'rounds':[],'tasks':[]}
            run['distributed_plan'] = plan
            progress(run)
        manifest = plan['original_bundle']
        path = pool.root(data)/'bundles'/(manifest['id']+'.json')
        if file_hash(path)!=manifest['sha256']:
            raise ValueError('Synchronized U-Net training input manifest changed')
        source = json.loads(path.read_text())
        if source['signature'] != pool.signature('unet'):
            raise ValueError('U-Net code or native package version changed. Restore the pinned runtime to recover this candidate.')
        objects = pool.root(data)/'objects'
        for oid, info in source['files'].items():
            control()
            if file_hash(objects/oid)!=info['sha256']:
                raise ValueError('Synchronized U-Net fitting source changed')
        original = source['request']
        options = original['options']
        # Keep original access paths for normal result review; verify the
        # immutable source objects and read only those copies during fitting.
        access = pool.replace_paths(original['access'],{name:str(objects/oid) for name,oid in source['bindings'].items()})
        options = pool.replace_paths(options,{name:str(objects/oid) for name,oid in source['bindings'].items()})
        labels = original['labels']
        heldout = [r for r in labels if r.get('role')=='validation' or r.get('benchmark_split') in ('validation','test')]
        import hashlib
        validation = {'sha256':hashlib.sha256(pool.canonical(heldout)).hexdigest()}
        folder.mkdir(parents=True,exist_ok=True)

        def fit(samples, classes, destination, previous):
            import joblib
            import torch

            torch.set_num_threads(2)
            bands, mean, std, shards, sizes = _prepare(samples, classes, workers, folder)
            reference = initial_state(len(bands),len(classes))
            epochs = options['epochs']
            weights = reference
            velocity = {name:np.zeros_like(value) for name,value in weights.items()}
            initialization = 'Seeded U-Net initialization'
            if previous and previous.is_file():
                saved = joblib.load(previous)
                if saved.get('model_id')=='unet' and saved.get('bands')==bands and saved.get('classes')==classes:
                    weights = {k:v.detach().cpu().numpy().copy() for k,v in saved['state_dict'].items()}
                    initialization = 'Continued compatible U-Net candidate with a fresh synchronized optimizer'
            completed = plan['rounds']
            for i, record in enumerate(completed,1):
                if record['round']!=i or file_hash(folder/f'round-{i:04d}.safetensors')!=record['checkpoint_sha256']:
                    raise ValueError('A previously committed U-Net barrier changed')
            if completed:
                weights,velocity = load_round(folder/f'round-{len(completed):04d}.safetensors',reference)
            for epoch in range(len(completed)+1,epochs+1):
                control()
                weights_file = folder/f'weights-{epoch:04d}.safetensors'
                _immutable_tensors(weights_file,weights)
                identifiers = _round_tasks(database,data,run,workers,weights_file,shards,sizes,bands,classes,epoch,validation)
                plan['tasks'] = identifiers
                run.update(distributed_task_ids=identifiers,stage=f'Waiting for U-Net round {epoch} of {epochs}')
                persist(run)
                try:
                    tasks = pool.wait_tasks(database,data,run,persist,identifiers,config,
                                            15+int(55*(epoch-1)/epochs),max(1,55//epochs),shutdown)
                except InterruptedError as exc:
                    raise mission_control.MissionStopped() from exc
                control()
                contributions, members = {}, []
                for rank,task in enumerate(tasks):
                    contributions[rank],member = _verified_gradient(database,data,task,
                        {'worker':workers[rank],'round':epoch,'rank':rank,'pixels':sizes[rank]},reference)
                    members.append(member)
                weights, velocity, loss = apply_round(weights,velocity,dict(enumerate(sizes)),contributions)
                barrier = folder/f'round-{epoch:04d}.safetensors'
                try:
                    save_round(folder,epoch,weights,velocity)
                except FileExistsError:
                    prior = load_round(barrier,reference)
                    if any(not np.array_equal(a[name],b[name]) for a,b in zip(prior,(weights,velocity)) for name in a):
                        raise ValueError('Recovered U-Net barrier differs from its verified gradients')
                completed.append({'round':epoch,'loss':loss,'members':members,'checkpoint_sha256':file_hash(barrier)})
                plan['tasks'] = []
                run.update(progress=15+int(55*epoch/epochs),stage=f'U-Net round {epoch} of {epochs} committed')
                progress(run)

            evidence = {'mode':'synchronized_unet','distinct_workers':len(workers),'rounds':completed,
                        'validation_manifest':validation,'training_pixels':sum(sizes),
                        'optimizer':'SGD with momentum; one pixel-weighted update per full fitting batch',
                        'partitioning':'Whole images when enough are available, otherwise contiguous fitting stripes with aligned context where supported'}
            run['distributed_training'] = evidence
            model = {'model_id':'unet','bands':bands,'classes':classes,'mean':mean,'std':std,
                     'state_dict':{name:torch.from_numpy(value.copy()) for name,value in weights.items()},
                     'training_pixels':sum(sizes),'implementation':'U-Net · synchronized gradients','initialization':initialization}
            tmp = destination.with_suffix('.partial')
            joblib.dump(model,tmp)
            os.replace(tmp,destination)
            return {'path':str(destination),'training_seconds':round(time.monotonic()-start,3),
                    'implementation':model['implementation'],'initialization':initialization,
                    'training_pixels':sum(sizes),'training_mode':'synchronized_unet',
                    'learning_curve':[{'epoch':r['round'],'loss':r['loss'],'scope':'full fitting batch'} for r in completed]}

        result = execute_job(run,original['images'],access,labels,data,progress,unet_trainer=fit,**options)
        if result['status']=='complete':
            return result
    except pool.CoordinatorRestart:
        run.update(status='queued',stage='Coordinator restarting; resuming from verified U-Net barrier')
    except (InterruptedError,mission_control.MissionStopped):
        run.update(status='cancelled',stage='Stopped; verified U-Net checkpoints retained')
    except Exception as exc:
        run.update(status='failed',stage='Synchronized U-Net needs attention',error=str(exc))
    if run['status']!='queued':
        for identifier in (plan or {}).get('tasks',[]):
            try:queue.control(identifier,'stop')
            except ValueError:pass
    persist(run)
    return run
