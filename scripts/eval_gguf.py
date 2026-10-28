import json,urllib.request,time,sys,os
port=sys.argv[1]; n=int(sys.argv[2]); out=sys.argv[3]
rows=[]
import glob
for f in sorted(glob.glob(os.path.join(os.environ.get('TEST_DIR', 'data/test'), '*.jsonl'))):
    rows+= [json.loads(l) for l in open(f)]
rows=rows[::max(1,len(rows)//n)][:n]
schema={"type":"object","properties":{"reply":{"type":"string"},"action":{"type":"object","properties":{"kind":{"type":"string"},"args":{"type":"object"}},"required":["kind","args"]},"mood_shift":{"type":"integer","enum":[-1,0,1]}},"required":["reply","action","mood_shift"]}
res_all=[];t0=time.time()
for r in rows:
    body={"messages":r["messages"][:-1],"temperature":0,"seed":1,"max_tokens":200,"chat_template_kwargs":{"enable_thinking":False},"json_schema":schema}
    res=json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900).read())
    c=res["choices"][0]["message"]["content"]
    g=json.loads(r["messages"][-1]["content"])
    try: d=json.loads(c); ok=True
    except Exception: d=None; ok=False
    res_all.append({"ok":ok,"kind":ok and d["action"]["kind"]==g["action"]["kind"],"mood":ok and d["mood_shift"]==g["mood_shift"],"reply":d["reply"] if ok else c,"gold":g["reply"]})
n=len(res_all)
print(json.dumps({"n":n,"json":sum(x["ok"] for x in res_all)/n,"acao":sum(x["kind"] for x in res_all)/n,"humor":sum(x["mood"] for x in res_all)/n,"seg":round(time.time()-t0)},ensure_ascii=False))
json.dump(res_all,open(out,"w"),ensure_ascii=False,indent=1)
