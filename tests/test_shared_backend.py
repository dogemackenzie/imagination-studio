"""Shared-backend decisions use measured peaks; cancellation never mutates its queue."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'spark'))
import shared_backend as B
import rtx_admit


def sample():
    return dict(host_available_mb=15000, host_total_mb=16000, vram_free_mib=12000, vram_known=True,
                cgroup=dict(known=True, cap=False), vram_reserve_mib=512)


def test_recorded_peaks_refuse_the_unsafe_start_even_above_the_floor():
    root=Path(__file__).resolve().parents[1] / 'docs/benchmarks'
    for file,weight in [('rtx3060-images.json',6617),('rtx3060-z-image-turbo.json',5914)]:
        p=json.loads((root/file).read_text())
        s=B.budgets(sample(),p,weight)
        assert s['need_host_mb'] > 7000
        assert B.decide(s)['code']=='insufficient_memory'
    assert s['need_host_mb'] + rtx_admit.HOST_FLOOR_MB > sample()['host_total_mb']


def test_missing_or_invalid_measurement_fails_closed():
    for p in [None,{}, {'host_available_mb': float('nan'), 'render':{}}]:
        assert B.decide(B.budgets(sample(),p,1))['code']=='unmeasured_configuration'


def test_margin_and_reserve_must_both_fit():
    profile=dict(host_available_mb=14000,vram_free_mib=12000,
                 render=dict(host_available_mb_min=12000,vram_free_mib_min=11000))
    s=B.budgets(sample(),profile,1000)
    assert B.decide(s)['ok']
    assert B.decide({**s,'host_available_mb':8000+2512-1})['code']=='insufficient_memory'
    assert B.decide({**s,'vram_free_mib':512+1256-1})['code']=='insufficient_vram'


def test_cancelled_client_retains_prompt_until_backend_finishes(tmp_path,monkeypatch):
    monkeypatch.setenv('STUDIO_HOME',str(tmp_path)); calls=[]; complete=False
    def exchange(method,path,payload=None,timeout=None):
        calls.append((method,path))
        if method=='POST':return {'prompt_id':'our-prompt'}
        return {'our-prompt':{'status':{'completed':complete}}}
    with B.session(exchange,'shared'):
        B.submit(exchange,'shared',{'prompt':{}})
    with pytest.raises(RuntimeError,match='may still be running'):
        with B.session(exchange,'shared'):pass
    complete=True
    with B.session(exchange,'shared'):pass
    assert not B.state_path().exists()
    assert calls==[('POST','/prompt'),('GET','/history/our-prompt'),('GET','/history/our-prompt')]


def test_lost_post_response_never_retries_automatically(tmp_path,monkeypatch):
    monkeypatch.setenv('STUDIO_HOME',str(tmp_path))
    def broken(*a,**k):raise RuntimeError('connection lost')
    with pytest.raises(RuntimeError,match='connection lost'):
        with B.session(broken,'shared'):B.submit(broken,'shared',{})
    with pytest.raises(RuntimeError,match='uncertain outcome'):
        with B.session(broken,'shared'):pass


def test_second_studio_client_cannot_submit(tmp_path,monkeypatch):
    monkeypatch.setenv('STUDIO_HOME',str(tmp_path))
    def no_calls(*a,**k):raise AssertionError('no network needed')
    with B.session(no_calls,'shared'):
        with pytest.raises(RuntimeError,match='Another Studio client'):
            with B.session(no_calls,'shared'):pass


def test_environment_cannot_lower_vram_reserve(monkeypatch):
    monkeypatch.setenv('RTX_VRAM_RESERVE_MIB','1')
    assert rtx_admit.vram_reserve_mib()==512


@pytest.mark.parametrize('model,size',[('sdxl_turbo',512),('z_image_turbo',1024)])
def test_complete_client_workflow_against_mock_backend(model,size,tmp_path,monkeypatch):
    import importlib,io,types
    from PIL import Image
    module=importlib.import_module(model)
    transport=module if model=='sdxl_turbo' else module.sdxl_turbo
    monkeypatch.setenv('STUDIO_HOME',str(tmp_path))
    monkeypatch.setattr(module,'offered',lambda:True)
    monkeypatch.setattr(module,'assess',lambda:dict(ok=True))
    monkeypatch.setattr(transport,'_sample_floor',lambda:{})
    buffer=io.BytesIO();Image.new('RGB',(size,size)).save(buffer,format='PNG')
    calls=[]
    def exchange(method,path,payload=None,timeout=30):
        calls.append((method,path))
        if path=='/queue':return {'queue_running':[],'queue_pending':[]}
        if path=='/prompt':
            assert payload['prompt']['4' if model=='z_image_turbo' else '2']['inputs']['text']=='a red door'
            return {'prompt_id':'own-job'}
        if path=='/history/own-job':return {'own-job':{'status':{'completed':True,'status_str':'success'},
            'outputs':{'save':{'images':[{'filename':'ours.png','type':'output'}]}}}}
        if path.startswith('/view?'):return buffer.getvalue()
        raise AssertionError(path)
    monkeypatch.setattr(transport,'exchange',exchange)
    args=types.SimpleNamespace(base=False,negative='',steps=None,width=size,height=size,seeds='',seed=1,output=str(tmp_path/'result.png'))
    rows=module.render_images(args,'a red door')
    assert rows[0]['model']==module.ID and rows[0]['width']==size
    assert (tmp_path/'result.png').is_file() and not B.state_path().exists()
    assert [p for m,p in calls if m=='POST']==['/prompt']


@pytest.mark.parametrize('model', ['sdxl_turbo', 'z_image_turbo'])
def test_queue_refuses_insufficient_headroom_before_any_submission(model, monkeypatch):
    import server
    from fastapi.testclient import TestClient
    module = getattr(server, model)
    monkeypatch.setattr(server.sdxl_turbo, 'submission', lambda *a: None)
    monkeypatch.setattr(server.z_image_turbo, 'submission', lambda *a: None)
    monkeypatch.setattr(module, 'submission', lambda *a: {'ok': True})
    monkeypatch.setattr(module, 'assess', lambda: {'ok': False, 'code': 'insufficient_memory',
                                                'reason': 'Measured peak plus reserve does not fit.'})
    before = set(server.jobs)
    reply = TestClient(server.app).post('/api/jobs', json={'kind': 'scout', 'params': {
        'image_model': module.ID, 'prompt': 'a door'}})
    assert reply.status_code == 400 and reply.json()['code'] == 'insufficient_memory'
    assert set(server.jobs) == before


def test_shared_container_shutdown_never_calls_backend_lifecycle(monkeypatch):
    import server
    monkeypatch.setenv('STUDIO_CONTAINER', '1')
    monkeypatch.setattr(server.subprocess, 'run', lambda *a, **k: pytest.fail('backend lifecycle called'))
    server.stop_session_comfy()


def test_backend_free_memory_cannot_override_a_lower_nvml_reading():
    assert B.shared_vram({**sample(), 'vram_free_mib': 100}, 12000 * 1024**2)['vram_free_mib'] == 100
    assert B.shared_vram({**sample(), 'vram_known': False}, 11000 * 1024**2)['vram_free_mib'] == 11000
    assert not B.shared_vram(sample(), -1)['vram_known']


def test_backup_restore_preserves_pending_backend_ownership(tmp_path, monkeypatch):
    import opsdata
    home = tmp_path / 'home'
    monkeypatch.setenv('STUDIO_HOME', str(home))
    B._save({'origin': 'shared', 'state': 'submitted', 'prompt_id': 'still-running'})
    opsdata.backup(home, tmp_path / 'backup')
    opsdata.restore(tmp_path / 'backup', tmp_path / 'restored')
    assert json.loads((tmp_path / 'restored/shared-prompt.json').read_text())['prompt_id'] == 'still-running'
