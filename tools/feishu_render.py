#!/usr/bin/env python3
"""飞书 Wiki 富文本渲染器：page spec(markdown 子集) → docx 原生块（标题/表格/callout/图片）。

用法：
  feishu_render.py render --node <token|url> --file spec.md --archive <归档路径> [--render-id <id>]
  feishu_render.py upload-image --node <token|url> --file x.png   # 单测传图

spec 语法：
  # / ## / ###     标题
  > 行（连续）      callout 结论卡
  | a | b |（连续） 表格，首行表头
  ![alt](path.png) 图片（先上传 drive media，再插 image 块）
  -  / 1.          无序/有序列表
  **bold**         行内加粗
  ---              分割线
  其余行           普通文本
"""
import argparse
import json
import mimetypes
import os
import re
import sys
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feishu_wiki import Feishu, node_token, archive_text, normalized_text, CONFIG, wiki_url

API = "https://open.feishu.cn/open-apis"


def _elements(text):
    """行内 **bold** 解析 → text_run elements。"""
    parts, out = re.split(r"(\*\*[^*]+\*\*)", text), []
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            out.append({"text_run": {"content": p[2:-2],
                                     "text_element_style": {"bold": True}}})
        else:
            out.append({"text_run": {"content": p}})
    return out or [{"text_run": {"content": " "}}]


class Spec:
    """把 page spec 解析成块树，返回 descendant API 可用的顶层/后代结构。"""

    def __init__(self, text):
        self.lines = text.splitlines()
        self.counter = 0

    def _bid(self):
        self.counter += 1
        return "b%d" % self.counter

    def _text(self, content, btype=2, field="text"):
        bid = self._bid()
        return bid, {"block_id": bid, "block_type": btype,
                     field: {"elements": _elements(content)}}

    def parse(self):
        """返回 (top_ids, descendants)。"""
        top, desc, i, n = [], [], 0, len(self.lines)
        while i < n:
            line = self.lines[i].rstrip()
            s = line.strip()
            if not s:
                i += 1
                continue
            if s == "---":
                bid = self._bid()
                top.append(bid)
                desc.append({"block_id": bid, "block_type": 22, "divider": {}})
                i += 1
                continue
            m = re.match(r"^(#{1,3})\s+(.*)", s)
            if m:
                level = len(m.group(1))
                bid, blk = self._text(m.group(2), 2 + level, "heading%d" % level)
                top.append(bid)
                desc.append(blk)
                i += 1
                continue
            if s.startswith(">"):
                # callout 块（19）在 descendant API 不可创建——用单列单元格"卡片"替代
                kids = []
                while i < n and self.lines[i].strip().startswith(">"):
                    cid, cblk = self._text(self.lines[i].strip()[1:].strip())
                    kids.append(cblk)
                    i += 1
                cell = self._bid()
                bid = self._bid()
                top.append(bid)
                desc.append({"block_id": bid, "block_type": 31,
                             "table": {"property": {"row_size": 1, "column_size": 1,
                                                    "column_width": [680],
                                                    "header_row": False}},
                             "children": [cell]})
                desc.append({"block_id": cell, "block_type": 32, "table_cell": {},
                             "children": [k["block_id"] for k in kids]})
                desc.extend(kids)
                continue
            if s.startswith("!["):
                m2 = re.match(r"!\[(.*?)\]\((.*?)\)", s)
                if not m2:
                    raise ValueError("图片语法错误：" + s)
                bid = self._bid()
                top.append(bid)
                desc.append({"block_id": bid, "block_type": 27,
                             "image": {"token": m2.group(2),
                                       "caption": m2.group(1) or ""}})
                i += 1
                continue
            if re.match(r"^\|.*\|", s):
                rows = []
                while i < n and re.match(r"^\|.*\|", self.lines[i].strip()):
                    cells = [c.strip() for c in self.lines[i].strip().strip("|").split("|")]
                    rows.append(cells)
                    i += 1
                # 跳过分隔行 |---|---|
                rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
                ncol = max(len(r) for r in rows)
                cell_ids, cell_blocks = [], []
                for r in rows:
                    for c in (r + [""] * ncol)[:ncol]:
                        tid, tblk = self._text(c)
                        cid = self._bid()
                        cell_ids.append(cid)
                        cell_blocks.append({"block_id": cid, "block_type": 32,
                                            "table_cell": {}, "children": [tid]})
                        cell_blocks.append(tblk)
                bid = self._bid()
                top.append(bid)
                width = max(70, 680 // ncol)
                desc.append({"block_id": bid, "block_type": 31,
                             "table": {"property": {"row_size": len(rows),
                                                    "column_size": ncol,
                                                    "column_width": [width] * ncol,
                                                    "header_row": True}},
                             "children": cell_ids})
                desc.extend(cell_blocks)
                continue
            m = re.match(r"^([-*])\s+(.*)", s)
            if m:
                bid, blk = self._text(m.group(2), 12, "bullet")
                top.append(bid)
                desc.append(blk)
                i += 1
                continue
            m = re.match(r"^(\d+)[.)]\s+(.*)", s)
            if m:
                bid, blk = self._text(m.group(2), 13, "ordered")
                top.append(bid)
                desc.append(blk)
                i += 1
                continue
            bid, blk = self._text(s)
            top.append(bid)
            desc.append(blk)
            i += 1
        if not top:
            raise ValueError("spec 没有产生任何块。")
        return top, desc


def upload_image(client, document_id, path):
    path = Path(path)
    data = path.read_bytes()
    boundary = uuid.uuid4().hex
    fields = {
        "file_name": path.name,
        "parent_type": "docx_image",
        "parent_node": document_id,
        "size": str(len(data)),
    }
    body = b""
    for k, v in fields.items():
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                 % (boundary, k, v)).encode()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    body += ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
             "Content-Type: %s\r\n\r\n" % (boundary, path.name, mime)).encode() + data + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode()
    req = Request(API + "/drive/v1/medias/upload_all", data=body, method="POST",
                  headers={"Authorization": "Bearer " + client.token,
                           "Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        body_txt = getattr(e, "read", lambda: b"")()
        raise RuntimeError("上传 HTTP 错误：%s %s" % (e, body_txt[:500]))
    if result.get("code") != 0:
        raise RuntimeError("图片上传失败：%s" % result.get("msg"))
    return result["data"]["file_token"]


def _png_size(path):
    """读 PNG 头拿像素尺寸；失败给 800x400 兜底。"""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            import struct
            w, h = struct.unpack(">II", head[16:24])
            return w, h
    except OSError:
        pass
    return 800, 400


def insert_image_block(client, document_id, pos, path, caption=""):
    """飞书插图三步：建空 image 块 → 素材上传(parent_node=块ID) → replace_image 回填。"""
    data = client.request(
        "POST", "/docx/v1/documents/{0}/blocks/{0}/children".format(document_id),
        {"children": [{"block_type": 27, "image": {}}], "index": pos},
        {"document_revision_id": -1, "client_token": str(uuid.uuid4())})
    block_id = data["data"]["children"][0]["block_id"]
    token = upload_image(client, block_id, path)
    client.request("PATCH", "/docx/v1/documents/{0}/blocks/{1}".format(document_id, block_id),
                   {"replace_image": {"token": token}}, {"document_revision_id": -1})
    return block_id


def insert_blocks(client, document_id, top, desc, chunk=40):
    """按序插入顶层块：连续非图片段走 descendant API；图片段走三步绑定。"""
    desc_by_id = {d["block_id"]: d for d in desc}
    pos = len(client.request("GET", "/docx/v1/documents/{0}/blocks/{0}".format(document_id),
                             query={"document_revision_id": -1})["data"]["block"].get("children", []))

    def subtree_of(bid):
        subtree, queue = [], [bid]
        while queue:
            b = desc_by_id[queue.pop(0)]
            subtree.append(b)
            queue.extend(b.get("children", []))
        return subtree

    def flush(group, group_desc):
        nonlocal pos
        if not group:
            return
        client.request("POST", "/docx/v1/documents/{0}/blocks/{0}/descendant".format(document_id),
                       {"index": pos, "children_id": group, "descendants": group_desc},
                       {"document_revision_id": -1})
        pos += len(group)

    group, group_desc = [], []
    for bid in top:
        blk = desc_by_id[bid]
        if blk["block_type"] == 27:
            flush(group, group_desc)
            group, group_desc = [], []
            img = blk["image"]
            insert_image_block(client, document_id, pos, img["token"], img.get("caption", ""))
            pos += 1
            continue
        st = subtree_of(bid)
        if group_desc and len(group_desc) + len(st) > chunk:
            flush(group, group_desc)
            group, group_desc = [], []
        group.append(bid)
        group_desc.extend(st)
    flush(group, group_desc)


def render_command(args):
    spec_text = Path(args.file).read_text(encoding="utf-8")
    client = Feishu()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    node = client.node(node_token(args.node))
    if node["obj_type"] != "docx":
        raise ValueError("仅支持 docx 节点。")
    doc = node["obj_token"]
    top, desc = Spec(spec_text).parse()
    # replace：归档旧文 → 删旧 children → 插新
    existing = client.read(node) if hasattr(client, "read") else None
    if existing is None:
        existing = client.request("GET", "/docx/v1/documents/%s/raw_content" % doc)["data"]["content"]
    if normalized_text(spec_text) in normalized_text(existing):
        return {"written": False, "reason": "内容与现页面一致", "url": wiki_url(config, node)}
    archive_text(args.archive, existing)
    root_path = "/docx/v1/documents/{0}/blocks/{0}".format(doc)
    root = client.request("GET", root_path, query={"document_revision_id": -1})["data"]["block"]
    old = len(root.get("children", []))
    while old:
        take = min(50, old)
        client.request("DELETE", root_path + "/children/batch_delete",
                       {"start_index": 0, "end_index": take},
                       {"document_revision_id": -1})
        old -= take
    insert_blocks(client, doc, top, desc)
    readback = client.request("GET", "/docx/v1/documents/%s/raw_content" % doc)["data"]["content"]
    probe = normalized_text(spec_text.splitlines()[max(0, len(spec_text.splitlines()) // 2)])
    if not normalized_text(readback):
        raise RuntimeError("渲染后回读为空，请核对页面。")
    return {"written": True, "blocks": len(desc), "removed": len(root.get("children", [])),
            "url": wiki_url(config, node)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    cmds = parser.add_subparsers(dest="command", required=True)
    r = cmds.add_parser("render", help="把 spec 渲染为 docx 原生块并整页替换（归档旧文）")
    r.add_argument("--node", required=True)
    r.add_argument("--file", required=True)
    r.add_argument("--archive", required=True)
    u = cmds.add_parser("upload-image", help="向文档上传图片，打印 file_token")
    u.add_argument("--node", required=True)
    u.add_argument("--file", required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    client = Feishu()
    node = client.node(node_token(args.node))
    if args.command == "upload-image":
        token = upload_image(client, node["obj_token"], args.file)
        print(json.dumps({"file_token": token}, ensure_ascii=False, indent=2))
        return
    if args.command == "render":
        print(json.dumps(render_command(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("错误: " + str(exc), file=sys.stderr)
        sys.exit(1)
