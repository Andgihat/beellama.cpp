"""Short live monitor contract test against an already running local service."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18200)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    base=f'http://127.0.0.1:{args.port}'
    def call(path,body=None):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req=urllib.request.Request(base+path, None if body is None else json.dumps(body).encode(),
                                   headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=120) as response:return json.load(response)
    deadline=time.monotonic()+120
    while True:
        try:
            health=call('/health')
            if health.get('loaded'):break
        except OSError:pass
        if time.monotonic()>deadline:raise RuntimeError('service did not become ready')
        time.sleep(.5)
    report={'status':'running','props':call('/props'),'samples':[]}
    report['props'].pop('chat_template',None)
    body={'messages':[{'role':'user','content':'Write a numbered list of 100 practical programming guidelines. Give one full sentence for each guideline; continue until all 100 are listed.'}],
          'temperature':0,'max_tokens':1024,'chat_template_kwargs':{'enable_thinking':False}}
    report['request']=body
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future=pool.submit(call,'/v1/chat/completions',body)
            printed=False
            while not future.done():
                started=time.monotonic();snap=call('/api/monitor/snapshot')
                snap['poll_ms']=(time.monotonic()-started)*1000
                report['samples'].append(snap)
                if not printed and any(r['decode_tps'] for r in snap['active']):
                    print(json.dumps({'live':snap['active'],'hardware':snap['hardware']}),flush=True)
                    printed=True
                time.sleep(.4)
            report['response']=future.result()
        final=call('/api/monitor/snapshot');report['final']=final
        active=[r for snap in report['samples'] for r in snap['active']]
        assert len({r['generated_tokens'] for r in active if r['decode_tps']})>1,'no live updates'
        assert max(s['poll_ms'] for s in report['samples'])<2000,'monitor blocked inference'
        assert not final['active'] and final['recent'][0]['status']=='completed'
        row=final['recent'][0]
        assert {r['id'] for r in active}=={row['id']}
        assert row['generated_tokens']==report['response']['usage']['completion_tokens']
        assert row['decode_tps']>0 and row['prefill_tps']>0
        assert final['hardware']['gpus'][0]['uuid']==report['props']['execution']['monitor_devices'][0]['uuid']
        assert final['sampled_at_ms']-final['hardware']['sampled_at_ms']<5000
        assert final['kvmem']['host_payload_bytes'] is not None
        # A normal continuation checks reuse reporting without a long-context quality experiment.
        follow={**body,'max_tokens':32,'messages':body['messages']+[report['response']['choices'][0]['message'],
                 {'role':'user','content':'Reply with exactly OK.'}]}
        report['follow_request']=follow;report['follow_response']=call('/v1/chat/completions',follow)
        report['follow_snapshot']=call('/api/monitor/snapshot')
        assert report['follow_snapshot']['recent'][0]['cached_tokens']>0
        report['status']='pass'
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'samples':len(report['samples']),'final':row,
                      'cache':final['kvmem']}),flush=True)


if __name__=='__main__':main()
