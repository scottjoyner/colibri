#!/usr/bin/env python3
"""Physical Joyner R9700 bounded Vulkan small-fixture pool A/B, no full GLM load."""
import hashlib,json,os,pathlib,re,statistics,subprocess,time,urllib.request
from datetime import datetime,timezone
root=pathlib.Path("/home/joyner/provider-r9700/colibri-pool-fixture-20261017/c")
out=pathlib.Path("/home/joyner/provider-r9700/colibri-pool-fixture-evidence-20261017")
out.mkdir(parents=True,exist_ok=True)
exe=root/"tests/test_vk_tier"
assert exe.is_file()
def health():
    for port in (18320,18321):
        with urllib.request.urlopen(f"http://100.83.215.83:{port}/health",timeout=4) as resp:
            if resp.status!=200:raise RuntimeError(f"port {port} unhealthy")
def psi():
    for line in pathlib.Path("/proc/pressure/memory").read_text().splitlines():
        if line.startswith("full "):
            return float(dict(x.split("=") for x in line.split()[1:])["avg10"])
    return None
# Each segment ABBA, followed by reverse order: four repetitions per mode.
order=[0,1,1,0,1,0,0,1]
results=[]
for index,pool in enumerate(order):
    health()
    before=psi()
    if before is not None and before>2:
        raise RuntimeError("NO_GO elevated memory pressure before trial")
    env=os.environ.copy()
    env.update({"COLI_VK_STAGED":"1","COLI_VK_UP_IMPORT":"1",
                "COLI_VK_UP_POOL":str(pool),"COLI_VK_TIER_FILL":"0","OMP_NUM_THREADS":"2"})
    start=time.perf_counter()
    p=subprocess.run([str(exe),"shaders/qmatmul.spv"],cwd=root,env=env,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=50)
    seconds=time.perf_counter()-start
    log=out/f"pooled_repeat_{index:02d}_{pool}.log"
    log.write_text(p.stdout)
    if p.returncode or not re.search(r"^PASS$",p.stdout,re.M):
        raise RuntimeError(f"fixture failed idx {index} pool={pool}")
    m=re.findall(r"reusable host imports: created (\d+), reused (\d+), fallback (\d+), peak ([0-9.]+) MiB",p.stdout)
    results.append({"trial":index,"pool_enabled":bool(pool),"wall_seconds":round(seconds,3),
                    "created_last":int(m[-1][0]) if m else 0,
                    "reused_last":int(m[-1][1]) if m else 0,
                    "fallback_last":int(m[-1][2]) if m else 0,
                    "peak_MiB":float(m[-1][3]) if m else 0,
                    "memory_full_psi_avg10_before":before,
                    "log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()})
    print("TRIAL",index,"POOL",pool,"SECONDS",round(seconds,3),flush=True)
    time.sleep(.3)
health()
off=[x["wall_seconds"] for x in results if not x["pool_enabled"]]
on=[x["wall_seconds"] for x in results if x["pool_enabled"]]
report={"schema":1,"date_utc":datetime.now(timezone.utc).isoformat(),"machine":"joyner",
        "driver":"RADV GFX1201 PCI 1002:7551","target":"small Vulkan tier fixture, not full GLM",
        "binary_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),
        "memory_limit_bytes":1073741824,"unit_swap_bytes":0,"production_unchanged":True,
        "trial_order":order,"trials":results,"median_baseline_seconds":round(statistics.median(off),3),
        "median_pool_seconds":round(statistics.median(on),3),
        "median_pool_wall_change_percent":round(100*(statistics.median(on)/statistics.median(off)-1),2),
        "source_pinned":"scottjoyner/colibri experiment/joyner-vulkan-import-reuse-pool-20261017",
        "notes":["test fixture covers small expert transfers, not GLM checkpoint",
                 "host import pool hard max 64MiB and 12 slots; primary device only",
                 "normal backend imports may differ in size, concurrency and memory pressure"]}
(out/"pooled_repeat_summary.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("median_baseline_seconds","median_pool_seconds","median_pool_wall_change_percent")}),flush=True)
