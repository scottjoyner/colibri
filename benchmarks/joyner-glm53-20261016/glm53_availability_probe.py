#!/usr/bin/env python3
"""Read-only, bounded GLM/fleet inference availability probe.

HTTP 200 is not a model-loaded or route-admission proof. The returned report
cannot authorize failover or dispatch; those decisions require the separately
authenticated admission/fencing control plane and a successful generation
witness. This probe deliberately sends only GET requests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import socket
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError

class _DenyRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None

_SAFE_HTTP=build_opener(_DenyRedirect())
def _safe_open(req,timeout:float):
    return _SAFE_HTTP.open(req,timeout=timeout)

MAX_RESPONSE=65536
ALLOWED_SCHEMES=frozenset({"http","https"})
STATES=frozenset({"UNREACHABLE","HEALTH_FAILED","HOST_HTTP_ONLY",
                  "MODEL_CATALOG_UNVERIFIED","MODEL_NOT_LISTED"})

def probe_json(url:str, *, timeout:float=2.0, opener=_safe_open)->tuple[str,dict|None]:
    """Classify transport failure separately from reachable but unhealthy HTTP."""
    if not (0.1<=timeout<=5.0):
        raise ValueError("timeout out of bounds")
    u=urlsplit(url)
    if (u.scheme not in ALLOWED_SCHEMES or not u.hostname or
        u.username or u.password or u.fragment or
        u.path not in ("/health","/v1/models","/route")):
        raise ValueError("unapproved probe URL")
    try:
        with opener(Request(url,headers={"Accept":"application/json"}),
                    timeout=timeout) as resp:
            if resp.status!=200:return "HTTP_FAILURE",None
            raw=resp.read(MAX_RESPONSE+1)
        if len(raw)>MAX_RESPONSE:return "INVALID_BODY",None
        data=json.loads(raw)
        return ("OK",data) if isinstance(data,dict) else ("INVALID_BODY",None)
    except HTTPError:
        return "HTTP_FAILURE",None
    except (TimeoutError,OSError):
        return "TRANSPORT_ERROR",None
    except (ValueError,TypeError,json.JSONDecodeError):
        return "INVALID_BODY",None

def get_json(url:str,*,timeout:float=2.0,opener=_safe_open):
    return probe_json(url,timeout=timeout,opener=opener)[1]

def model_ids(catalog:dict)->frozenset[str]:
    items=catalog.get("data")
    if not isinstance(items,list):
        return frozenset()
    return frozenset(
        m["id"] for m in items if isinstance(m,dict)
        and type(m.get("id")) is str and 0<len(m["id"])<=200
    )

def observe(endpoint:str,expected_model:str,*,opener=_safe_open)->dict:
    u=urlsplit(endpoint)
    if (u.scheme not in ALLOWED_SCHEMES or not u.hostname or
        u.username or u.password or u.path not in ("","/") or
        u.query or u.fragment or not expected_model or len(expected_model)>200):
        raise ValueError("invalid endpoint or model identity")
    root=endpoint.rstrip("/")
    health_status,health=probe_json(root+"/health",opener=opener)
    if health_status=="OK" and (
        health.get("healthy") is False or
        health.get("status") in ("failed","unhealthy","error","down")):
        health_status="REPORTED_UNHEALTHY"
    if health_status!="OK":
        return {"state":"UNREACHABLE" if health_status=="TRANSPORT_ERROR"
                else "HEALTH_FAILED","http_health":False,
                "probe_failure":health_status,
                "catalog_lists_model":False,"model_loaded_proven":False,
                "routable":False}
    catalog=get_json(root+"/v1/models",opener=opener)
    if catalog is None:
        state="HOST_HTTP_ONLY"
        listed=False
    else:
        listed=expected_model in model_ids(catalog)
        state="MODEL_CATALOG_UNVERIFIED" if listed else "MODEL_NOT_LISTED"
    return {"state":state,"http_health":True,"catalog_lists_model":listed,
            "model_loaded_proven":False,"routable":False}

def fleet_observation(local_health:dict|None,route:dict|None,
                      remote:dict)->dict:
    host_healthy=bool(isinstance(local_health,dict) and
                      local_health.get("healthy") is True)
    # Local node healthy and a list of HTTP links cannot admit a remote GLM.
    advertised=(route or {}).get("default_route") if isinstance(route,dict) else None
    result={"observed_at_utc":datetime.now(timezone.utc).isoformat(),
            "local_host_healthy":host_healthy,
            "local_gateway_advertised_route":bool(advertised),
            "remote_state":remote["state"],
            "model_loaded_verified":False,
            "eligible_for_dispatch":False,
            "failover_authorized":False,
            "decision":"HOLD",
            "blockers":[]}
    if remote["state"]=="UNREACHABLE":
        result["blockers"].append("remote inference endpoint unreachable")
    elif remote["state"]=="HEALTH_FAILED":
        result["blockers"].append("remote endpoint reachable but health response invalid or unhealthy")
    elif remote["state"]=="MODEL_CATALOG_UNVERIFIED":
        result["blockers"].append("catalog membership is not generation-proof inference")
    else:
        result["blockers"].append("no verified model-specific generation response")
    if not advertised:
        result["blockers"].append("local gateway has no default inference route")
    result["blockers"].append("operator-signed model admission, owner-bound fencing and accepted fallback missing")
    return result

def main():
    ap=argparse.ArgumentParser(description="Read-only GET-only model readiness inspection")
    ap.add_argument("--inference-base",required=True)
    ap.add_argument("--expected-model",required=True)
    ap.add_argument("--local-health-base",default="http://127.0.0.1:8899")
    a=ap.parse_args()
    local_root=a.local_health_base.rstrip("/")
    remote=observe(a.inference_base,a.expected_model)
    report=fleet_observation(get_json(local_root+"/health"),
                             get_json(local_root+"/route"),remote)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0
if __name__=="__main__":
    main()
