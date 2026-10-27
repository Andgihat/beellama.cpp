"""Model-free invariants for PolyStrata's unified monitor."""
import collections
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'backends/polystrata'))
from serve.kvmem_monitor import StorageLog, execution, snapshot


class MonitorTest(unittest.TestCase):
    def service(self):
        now = time.time()
        return NS(started_at=now-10, status_lock=threading.Lock(), live_reqs={}, status={'busy':False},
                  rate=[], history=[], totals={'requests':0}, kvmem_monitor_gpu_uuid='GPU-test',
                  kvmem_monitor_log=StorageLog(), engine=NS(batch=0, info={'kvmem_host_mib':256,'ssd':0}),
                  telemetry=NS(snapshot=lambda: {'now':{'gpu_util':42,'ram_used':123},
                     'static':{'gpu_name':'test GPU'},'sampled_at_ms':(now-8)*1000}),
                  _rate_of=lambda st, rate: rate[-1][1])

    def test_lifecycle_parallel_rates_identity_privacy_and_missing(self):
        s = self.service();now=time.time()
        a = {'busy':True,'started':now-1,'first_token':now-.5,'generated':12,
             'prompt_tokens':100,'max_tokens':50,'kvmem_monitor_id':'a','tail':'SECRET'}
        b = dict(a, kvmem_monitor_id='b',generated=25)
        s.live_reqs={1:(a,[(now,8)]),2:(b,[(now,17)])};s.engine.batch=2
        with patch('serve.kvmem_monitor.time.time', return_value=now):
            x=snapshot(s)
        self.assertEqual([r['decode_tps'] for r in x['active']],[8,17])
        self.assertEqual([r['decode_tps_mean'] for r in x['active']],[24,50])
        self.assertEqual([r['id'] for r in x['active']],['a','b'])
        self.assertNotIn('SECRET',str(x))
        self.assertIsNone(x['kvmem']['retained_sessions'])
        self.assertIsNone(x['kvmem']['host_payload_bytes'])
        self.assertLess(x['hardware']['sampled_at_ms'],x['sampled_at_ms']-5000)
        self.assertEqual(execution(s)['monitor_devices'],[{'uuid':'GPU-test'}])
        s.live_reqs={};s.status=a;s.rate=[(now,8)];s.engine.batch=0
        s.engine.reused=80;s.engine.prefill_tok_s_mean=250
        row=snapshot(s)['active'][0]
        self.assertEqual(row['cached_tokens'],80)
        self.assertEqual(row['prefill_tps'],250)
        s.status={'busy':False}
        s.live_reqs={};s.history=[{'time':now-1,'duration_s':1,'finish':'disconnect',
            'output_tokens':1,'decode_tok_s':1000000,'kvmem_monitor_id':'a',
            'prompt_tokens':100,'reused':90,'prompt_read':10,'prompt_ms':20}]
        x=snapshot(s)
        self.assertFalse(x['active']);self.assertEqual(x['recent'][0]['id'],'a')
        self.assertEqual(x['recent'][0]['decode_tps'],0)
        self.assertEqual(x['recent'][0]['decode_tps_mean'],0)
        self.assertEqual(x['recent'][0]['status'],'disconnected')
        self.assertEqual(x['recent'][0]['prefill_tps'],500)

    def test_log_partial_line_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native.log';path.write_bytes(b'KVMEM_STORAGE host_bytes=42\nKVMEM_SESSIONS parked=2')
            engine=NS(log_path=str(path),log_start=0);log=StorageLog()
            self.assertEqual(log.poll(engine),{'host_bytes':42})
            with path.open('ab') as stream:stream.write(b' restores=3\n')
            self.assertEqual(log.poll(engine)['parked'],2)
            engine.log_start=path.stat().st_size
            self.assertEqual(log.poll(engine),{})
            path.write_bytes(b'KVMEM_STORAGE host_bytes=1\n')
            self.assertEqual(log.poll(engine),{'host_bytes':1})


if __name__ == '__main__':unittest.main()
