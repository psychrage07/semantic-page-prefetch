"""End-to-end check of real idle jobs through the local frontend/API proxy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import local_lab


def verify(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    local_lab.LAB=out/"jobs";local_lab.LAB.mkdir()
    local_lab.JOBS={}
    server=local_lab.ThreadingHTTPServer(("127.0.0.1",args.api_port),local_lab.Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f"http://127.0.0.1:{args.frontend_port}"
    def request(path,payload=None):
        headers={"Content-Type":"application/json"} if payload is not None else {}
        req=Request(base+path,data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
        with urlopen(req,timeout=15) as response:return response.read()
    def api(path,payload=None):return json.loads(request(path,payload))
    saved=[]
    try:
        assert b"Test idle neural hybrid" in request("/?view=lab")
        assert api("/api/health")["ok"]
        for dataset in ("small","large"):
            assert api(f"/api/idle-hybrid?dataset={dataset}")["available"]
            job=api("/api/jobs",{"preset":"idle-hybrid","dataset":dataset,"queries":10})["job"]
            print(f"Started {dataset} real API job {job['id']}",flush=True)
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                response=api(f"/api/jobs/{job['id']}");job=response["job"]
                if job["status"] not in ("running","cancelling"):break
                time.sleep(1)
            assert job["status"]=="completed",response
            content=api(f"/api/file?path={job['root']}/idle-result.json")["content"]
            result=json.loads(content)
            assert result["artifact_type"]=="idle_hybrid_live" and result["sql_results_identical"]
            assert len(result["rows"])==6 and {r["policy"] for r in result["rows"]}=={"P0","P2","P3","P4-original","P4-cached","P4-selective"}
            assert result["dataset"]==dataset and result["queries_per_policy"]==10
            assert json.loads(request(f"/api/download?path={job['root']}/idle-result.json"))==result
            for row in result["rows"]:
                assert row["runtime"]["worker_shutdown_clean"] and not row["runtime"]["worker_alive"]
                assert row["protected_artifacts_unchanged"]
                for name in ("trace.jsonl","queries.csv","runtime_stats.json"):
                    assert (Path(row["output"])/name).is_file()
            saved.append({"dataset":dataset,"job_id":job["id"],"result":str(local_lab.BACKEND/job["root"]/"idle-result.json"),"policies":6})
            print(f"Verified {dataset}: 60 real SQL executions, downloads and trace artifacts.",flush=True)
        job=api("/api/jobs",{"preset":"idle-hybrid","dataset":"small","queries":30})["job"]
        api(f"/api/jobs/{job['id']}/cancel",{})
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            job=api(f"/api/jobs/{job['id']}")["job"]
            if job["status"]=="cancelled":break
            time.sleep(.25)
        assert job["status"]=="cancelled"
        for path,payload in (("/api/idle-hybrid?dataset=outside",None),("/api/jobs",{"preset":"idle-hybrid","queries":101})):
            try:request(path,payload)
            except HTTPError as error:assert error.code==400
            else:raise AssertionError("Invalid request was accepted")
        verification={"frontend_http_ok":True,"frontend_api_proxy_ok":True,"small_and_large_live_jobs_passed":True,
            "query_executions":120,"download_checks_passed":True,"cancellation_passed":True,"invalid_requests_rejected":True,
            "browser_visual_check":"Unavailable: CUA inventory has no browser and the in-app browser is unavailable.","jobs":saved}
        (out/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
        print(json.dumps(verification,indent=2),flush=True)
    finally:
        proc=local_lab.ACTIVE["process"]
        if proc and proc.poll() is None:local_lab._stop_process(proc)
        server.shutdown();server.server_close();thread.join(5)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--api-port",type=int,default=18765)
    parser.add_argument("--frontend-port",type=int,default=14173)
    verify(parser.parse_args())
