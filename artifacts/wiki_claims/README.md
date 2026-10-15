# artifacts/wiki_claims — 执行认领登记

多 agent / 多线程并行时的执行认领区。**本目录除本 README 外不入 git**（`.gitignore` 已排除）。

## 规则

- 认领一个工作线程前，先在本目录建一份 claim 文件，文件名 `<topic>_<operator>.json`；
- 同 `<topic>` 已有未释放（`status` 为 `active`）的 claim 时，不得重复认领；
- claims 是**执行认领**：登记的是"谁正在做哪件事"，历史补登（追溯已完成工作）不进 claims，走 `ledger/` + `docs/experiments/`；
- 工作结束把 `status` 改为 `released` 并保留文件备查（本目录运行时文件不入 git，仅以磁盘为准）。

## claim 文件格式

```json
{
  "topic": "<工作主题代号，如 scene_joint_sig17>",
  "operator": "<agent/操作者代号，机器用代号 34/73>",
  "branch": "<git 分支>",
  "claimed_at": "<ISO8601>",
  "released_at": null,
  "owned_paths": ["<本次认领负责的路径集合>"],
  "status": "active | released",
  "note": "<一句话说明认领范围>"
}
```
