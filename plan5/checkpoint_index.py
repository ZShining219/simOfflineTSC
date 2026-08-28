"""Atomic checkpoint index with immutable path/hash entries."""
from pathlib import Path
from sequential.io import atomic_json, read_json, sha256_file


class CheckpointIndex:
    def __init__(self, path):
        self.path=Path(path)
        if self.path.exists(): self.payload=read_json(self.path)
        else: self.payload={"schema_version":1,"checkpoints":[]}

    def register(self, *, logical_run_id, checkpoint_type, episode, path, canonical_state_digest):
        path=Path(path)
        if not path.is_file(): raise FileNotFoundError(path)
        digest=sha256_file(path)
        entry={"logical_run_id":logical_run_id,"checkpoint_type":checkpoint_type,"episode":int(episode),"path":str(path.resolve()),"sha256":digest,"canonical_state_digest":canonical_state_digest}
        for existing in self.payload["checkpoints"]:
            if (existing["logical_run_id"],existing["checkpoint_type"],existing["episode"]) == (logical_run_id,checkpoint_type,int(episode)):
                if existing != entry: raise ValueError("Checkpoint index identity collision")
                return existing
        self.payload["checkpoints"].append(entry); self.payload["checkpoints"].sort(key=lambda x:(x["logical_run_id"],x["checkpoint_type"],x["episode"]))
        atomic_json(self.path,self.payload); return entry

    def validate(self):
        for item in self.payload.get("checkpoints",[]):
            if not Path(item["path"]).is_file() or sha256_file(item["path"]) != item["sha256"]: raise ValueError("Checkpoint index hash/path mismatch")
        return {"valid":True,"count":len(self.payload.get("checkpoints",[]))}
