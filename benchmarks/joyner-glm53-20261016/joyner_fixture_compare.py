#!/usr/bin/env python3
"""Full fixture A/B against unchanged binary, no live service side effects."""
import hashlib,json,os,pathlib,subprocess,time,sys
base=pathlib.Path("/home/joyner/provider-r9700/colibri-glm53-eval")
new=pathlib.Path("/home/joyner/provider-r9700/colibri-fork-glm53-bounded-20261016")
fixtures=[base/"fixtures/glm53_stream-i4",base/"fixtures/glm53_stream-deq",base/"fixtures/glm53_mm_tiny"]
receipt=new/"benchmarks/joyner-bounded-fixture-20261016.json"
results=[]
for fixture in fixtures:
    if not (fixture/"ref.json").is_file(): continue
    ref=json.loads((fixture/"ref.json").read_text())
    tokens=ref.get("prompt_ids") or ref.get("prompt")
    if not tokens:continue
    args=["--model",str(fixture),"--ids",",".join(map(str,tokens)),"--greedy","3","--logits"]
    for bits in (4,8):
        runs=[]
        for flavor,binary,on in [("baseline",base/"c/glm53",False),("bounded",new/"c/glm53",True)]:
            env=dict(os.environ,GLM53_BITS=str(bits),GLM53_BOUNDED_DENSE="1" if on else "0",
                     GLM53_DENSE_SPILL_DIR=sys.argv[1] if on else "",
                     COLI_VULKAN="0",GLM53_EXPERT_GB="0.03",OMP_NUM_THREADS="3")
            t=time.monotonic()
            proc=subprocess.run([str(binary),*args],cwd=base/"c",capture_output=True,text=True,env=env,timeout=85)
            if proc.returncode:
                raise RuntimeError(f"{flavor} {fixture.name} bits{bits} rc={proc.returncode}: {proc.stderr[-600:]}")
            lines={}
            for line in proc.stdout.splitlines():
                words=line.split()
                if words and words[0] in ("teacher_forcing","greedy","last_logits"):
                    lines[words[0]]=words[1:]
            if set(lines)!={"teacher_forcing","greedy","last_logits"}:
                raise RuntimeError(f"{flavor} {fixture.name} missing output: {sorted(lines)}")
            runs.append({"flavor":flavor,"model":fixture.name,"bits":bits,
                         "elapsed_s":round(time.monotonic()-t,3),"hashes":{k:hashlib.sha256(" ".join(v).encode()).hexdigest() for k,v in lines.items()},
                         "logit_size":len(lines["last_logits"]),"data":lines})
        agree={k:runs[0]["data"][k]==runs[1]["data"][k] for k in ("teacher_forcing","greedy","last_logits")}
        if not all(agree.values()):
            raise RuntimeError(f"NUMERICAL MISMATCH {fixture.name} bits={bits}: {agree}")
        print("PASS",fixture.name,"bits",bits,"teacher_forcing,greedy,last_logits identical",
              "baseline_s",runs[0]["elapsed_s"],"bounded_s",runs[1]["elapsed_s"],flush=True)
        results.append({"fixture":fixture.name,"bits":bits,"agreement":agree,
                       "baseline_elapsed_s":runs[0]["elapsed_s"],"bounded_elapsed_s":runs[1]["elapsed_s"],
                       "sha256":runs[0]["hashes"]})
if not results:raise RuntimeError("NO FIXTURES: refuse empty success")
receipt.write_text(json.dumps({"branch_commit":"abf2aad","host":"joyner","kind":"CPU fixture A/B",
                               "model":"GLM-5.3 synthetic fixture, not real checkpoint",
                               "total_cases":len(results),"results":results},indent=2)+"\n")
print("PASS fixture comparisons",len(results),"receipt",receipt)
