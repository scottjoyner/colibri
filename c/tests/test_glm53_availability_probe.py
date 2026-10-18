#!/usr/bin/env python3
"""One node HTTP OK can never auto-admit an inference route."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

P=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/glm53_availability_probe.py"
spec=importlib.util.spec_from_file_location("availability_probe_test",P)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class Response:
    def __init__(self,obj,code=200):
        self.body=json.dumps(obj).encode() if not isinstance(obj,bytes) else obj
        self.status=code
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def read(self,n=-1):return self.body[:n]

def fake(health,models,code=200):
    requested=[]
    def opener(req,timeout):
        requested.append((req.get_method(),req.full_url,timeout))
        if req.full_url.endswith("/health"):return Response(health,code)
        if req.full_url.endswith("/v1/models"):return Response(models)
        raise AssertionError("unexpected endpoint")
    return opener,requested

class ReadinessTests(unittest.TestCase):
    def test_host_http_not_loaded(self):
        opener,req=fake({"status":"ok"},None)
        state=m.observe("http://127.0.0.1:18320","glm53",opener=opener)
        self.assertEqual(state["state"],"HOST_HTTP_ONLY")
        self.assertFalse(state["routable"])
        self.assertTrue(all(q[0]=="GET" for q in req))
    def test_model_catalog_not_inference_proof(self):
        opener,req=fake({"status":"ok"},{"data":[{"id":"glm53"}]})
        state=m.observe("http://127.0.0.1:18320","glm53",opener=opener)
        self.assertEqual(state["state"],"MODEL_CATALOG_UNVERIFIED")
        self.assertFalse(state["model_loaded_proven"])
        self.assertFalse(state["routable"])
    def test_catalog_wrong_model(self):
        opener,_=fake({"status":"ok"},{"data":[{"id":"minicpm"}]})
        self.assertEqual(m.observe("http://127.0.0.1:18320","glm53",opener=opener)["state"],
                         "MODEL_NOT_LISTED")
    def test_network_loss(self):
        def fail(*a,**kw):raise TimeoutError("no route")
        state=m.observe("http://127.0.0.1:18320","glm53",opener=fail)
        self.assertEqual(state["state"],"UNREACHABLE")
        self.assertFalse(state["routable"])
    def test_explicit_unhealthy_http200_denied(self):
        opener,_=fake({"healthy":False},{"data":[{"id":"glm53"}]})
        state=m.observe("http://127.0.0.1:18320","glm53",opener=opener)
        self.assertEqual(state["state"],"HEALTH_FAILED")
        self.assertEqual(state["probe_failure"],"REPORTED_UNHEALTHY")
    def test_http_redirect_denied_without_following(self):
        from urllib.error import HTTPError
        def redirect(req,timeout):
            raise HTTPError(req.full_url,302,"Location",{"Location":"http://example.invalid"},None)
        status,data=m.probe_json("http://127.0.0.1:18320/health",opener=redirect)
        self.assertEqual(status,"HTTP_FAILURE")
        self.assertIsNone(data)
    def test_non200_denied_and_not_confused_with_network_failure(self):
        opener,_=fake({"status":"failed"},{"data":[{"id":"glm53"}]},code=503)
        state=m.observe("http://127.0.0.1:18320","glm53",opener=opener)
        self.assertEqual(state["state"],"HEALTH_FAILED")
        self.assertEqual(state["probe_failure"],"HTTP_FAILURE")
    def test_malformed_health_is_not_network_loss(self):
        opener,_=fake(b"{invalid",{"data":[{"id":"glm53"}]})
        self.assertEqual(m.observe("http://127.0.0.1:18320","glm53",opener=opener)["state"],
                         "HEALTH_FAILED")
    def test_malformed_catalog_denied(self):
        opener,_=fake({"status":"ok"},b"{broken")
        self.assertEqual(m.observe("http://127.0.0.1:18320","glm53",opener=opener)["state"],
                         "HOST_HTTP_ONLY")
    def test_oversize_catalog_denied(self):
        opener,_=fake({"status":"ok"},b" "*65537)
        self.assertEqual(m.observe("http://127.0.0.1:18320","glm53",opener=opener)["state"],
                         "HOST_HTTP_ONLY")
    def test_gateway_false_health_positive_cannot_route(self):
        remote={"state":"UNREACHABLE","routable":False}
        result=m.fleet_observation({"healthy":True,"default_route":""},
                                   {"default_route":""},remote)
        self.assertTrue(result["local_host_healthy"])
        self.assertFalse(result["eligible_for_dispatch"])
        self.assertFalse(result["failover_authorized"])
        self.assertEqual(result["decision"],"HOLD")
    def test_gateway_route_advertisement_still_not_authorized(self):
        remote={"state":"MODEL_CATALOG_UNVERIFIED","routable":False}
        result=m.fleet_observation({"healthy":True},
                                   {"default_route":"http://localhost:18320"},remote)
        self.assertTrue(result["local_gateway_advertised_route"])
        self.assertFalse(result["failover_authorized"])
    def test_invalid_url_rejected(self):
        for x in ("file:///etc/passwd","http://a:1/credentials",
                  "http://user:pass@127.0.0.1:1",
                  "http://127.0.0.1:18320?q=1"):
            with self.subTest(url=x),self.assertRaises(ValueError):
                m.observe(x,"glm53")
    def test_invalid_probe_path_rejected(self):
        with self.assertRaises(ValueError):
            m.get_json("http://127.0.0.1:8899/token")
    def test_invalid_timeout_rejected(self):
        with self.assertRaises(ValueError):
            m.get_json("http://127.0.0.1:8899/health",timeout=50)
    def test_bad_catalog_type_rejected(self):
        self.assertEqual(m.model_ids({"data":{"id":"glm53"}}),frozenset())
    def test_host_health_does_not_imply_route(self):
        r=m.fleet_observation({"healthy":True},{"routes":[{"name":"health"}]},
                              {"state":"MODEL_NOT_LISTED","routable":False})
        self.assertFalse(r["eligible_for_dispatch"])
        self.assertFalse(r["local_gateway_advertised_route"])
if __name__=="__main__":
    unittest.main()
