"""Short streaming regression for sparse attention samples and thinking controls.

Run against an already loaded loopback service with solo MTP enabled. Does not
change server settings or stop the user's service; saves every test request/event.
"""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18200)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base=f'http://127.0.0.1:{args.port}'
    def get(path):
        with opener.open(base+path,timeout=10) as response:return json.load(response)
    result={'status':'running','props':get('/props'),'cases':[]}
    result['props'].pop('chat_template',None)
    cases=[
        ('thinking_budget',{'reasoning_effort':'medium','reasoning_budget_tokens':64},
         'Work out the sum of all integers from 1 to 100. Think briefly, then answer in one sentence.'),
        ('thinking_off',{'chat_template_kwargs':{'enable_thinking':False}},
         'Explain the merge sort recurrence and give a small worked example. Use at most 100 words.'),
        ('thinking_on',{'chat_template_kwargs':{'enable_thinking':True},'reasoning_budget_tokens':96},
         'A rectangle has perimeter 30 and area 54. Work out its side lengths. Think briefly, then give the answer.')]
    try:
        for name,options,prompt in cases:
            body={'model':'kvmem-polystrata-qwen35','messages':[{'role':'user','content':prompt}],
                  'temperature':0,'max_tokens':384,'stream':True,**options}
            case={'name':name,'request':body,'events':[],'done':False}
            result['cases'].append(case)
            request=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),
                                           headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=90) as response:
                for raw in response:
                    line=raw.decode().strip()
                    if not line.startswith('data:'):continue
                    data=line[5:].strip()
                    if data=='[DONE]':case['done']=True;break
                    event=json.loads(data);case['events'].append(event)
                    assert 'error' not in event,event
            choices=[c for event in case['events'] for c in event.get('choices',[])]
            case['answer']=''.join(c.get('delta',{}).get('content') or '' for c in choices)
            case['reasoning']=''.join(c.get('delta',{}).get('reasoning_content') or '' for c in choices)
            case['finish']=[c['finish_reason'] for c in choices if c.get('finish_reason')]
            assert case['done'] and case['finish'] and case['answer'],case
            assert all(f in ('stop','length') for f in case['finish']),case['finish']
            if name=='thinking_off':assert not case['reasoning'],case
            else:assert case['reasoning'] and case['finish']==['stop'],case
            print(json.dumps({k:case[k] for k in ('name','finish','done','answer')}),flush=True)
        result['status']='pass'
    finally:
        result['metrics']=get('/metrics')
        result['metrics'].pop('history',None)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
