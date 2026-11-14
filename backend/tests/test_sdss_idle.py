import hashlib
import json
import sqlite3

import pytest
import torch

from sdss_idle_workload import bucket, make_cases, signature
from sdss_idle_experiment import apply_frozen_neural_weights, sha, tensor_hash
from test_idle_neural import examples, make_model


def test_sdss_workload_is_disjoint_by_parameter_signature_and_read_only(tmp_path):
    database = tmp_path / "astronomy.sqlite"
    with sqlite3.connect(database) as con:
        con.executescript("CREATE TABLE PhotoObj(objID INTEGER PRIMARY KEY,ra REAL,dec REAL,u REAL,g REAL,r REAL,i REAL,z REAL,type INTEGER,run INTEGER,field INTEGER); CREATE TABLE SpecObj(specObjID INTEGER PRIMARY KEY,bestObjID INTEGER,plate INTEGER,redshift REAL,specClass INTEGER);")
        con.executemany("INSERT INTO PhotoObj VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(587722982278693005+i,i*.1,i*.01,17.,16.,15.+i*.01,14.,13.,3,1+i%4,1+i%20) for i in range(128)])
        con.executemany("INSERT INTO SpecObj VALUES (?,?,?,?,?)",[(105212596461764608+i,587722982278693005+i,1+i%8,i*.001,2) for i in range(128)])
    before = sha(database); seen = set(); partitions = []
    for name, allowed in (("training",set(range(7))),("validation",{7}),("test",{8,9})):
        cases = make_cases(database, 7701, 12, name, seen)
        keys = {signature(c.template_id,c.params) for c in cases}
        assert len(keys) == len(cases) and all(bucket(k) in allowed for k in keys)
        assert all(not keys.intersection(previous) for previous in partitions)
        partitions.append(keys)
        with sqlite3.connect(database.as_uri()+"?mode=ro",uri=True) as con:
            for case in cases:
                result = list(con.execute(case.sql,case.params))
                if case.template_id == "sdss_object":
                    assert result[0][0] == case.params[0] and isinstance(case.params[0],int)
                    assert case.params[0] > 2**53
    assert sha(database) == before


def test_transfer_uses_frozen_source_weights_without_replacing_target_statistics(tmp_path):
    torch.set_num_threads(1)
    data = examples(); source = make_model(data); target = make_model(data)
    source.fit(data,data,epochs=1,budget=1);target.fit(data,data,epochs=1,budget=1)
    with torch.no_grad(): source.model.net[-1].bias.add_(1)
    checkpoint = tmp_path / "source.pt"
    torch.save({"input_dim":source.model.net[0].in_features,"config":{"hidden":source.config.hidden},
                "state_dict":source.model.state_dict()},checkpoint)
    prior_before = json.dumps(target.prior.counts,sort_keys=True)
    before = sha(checkpoint)
    result = apply_frozen_neural_weights(target,checkpoint,before)
    assert result == tensor_hash(source.model) == tensor_hash(target.model)
    assert not target.model.training and all(not p.requires_grad for p in target.model.parameters())
    assert json.dumps(target.prior.counts,sort_keys=True) == prior_before
    assert sha(checkpoint) == before
    with pytest.raises(ValueError,match="checkpoint changed"):
        apply_frozen_neural_weights(target,checkpoint,"invalid")
