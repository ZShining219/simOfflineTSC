#!/usr/bin/env python3
"""项目飞书 Wiki 入口：自建应用身份、标准库 HTTP 客户端，无仿真器依赖。"""

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import uuid


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/feishu.json"
CREDENTIALS = ROOT / ".secrets/feishu.json"
API = "https://open.feishu.cn/open-apis"
STATUSES = ("草案", "待执行", "执行中", "受阻", "已完成", "已停止")


def configure():
    """凭证仅交互输入，避免进入命令行参数及 shell 历史。"""
    app_id = input("App ID: ").strip()
    app_secret = getpass.getpass("App Secret（输入不显示）: ").strip()
    if not app_id.startswith("cli_") or not app_secret:
        raise ValueError("App ID 应以 cli_ 开头，App Secret 不能为空。")
    CREDENTIALS.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".feishu-", dir=CREDENTIALS.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"app_id": app_id, "app_secret": app_secret}, stream)
            stream.write("\n")
        os.replace(name, CREDENTIALS)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print("凭证已保存到 .secrets/feishu.json（权限 600，Git 已忽略）。")


class Feishu:
    def __init__(self):
        env_id = os.environ.get("FEISHU_APP_ID", "").strip()
        env_secret = os.environ.get("FEISHU_APP_SECRET", "").strip()
        if env_id or env_secret:
            credentials = {"app_id": env_id, "app_secret": env_secret}
        elif CREDENTIALS.exists():
            credentials = json.loads(CREDENTIALS.read_text(encoding="utf-8"))
        else:
            raise ValueError("请先执行 python tools/feishu_wiki.py configure。")
        if not credentials.get("app_id") or not credentials.get("app_secret"):
            raise ValueError("App ID 和 App Secret 必须成对配置。")
        self.redactions = [credentials["app_secret"]]
        self.token = None
        response = self.request("POST", "/auth/v3/tenant_access_token/internal", credentials)
        self.token = response["tenant_access_token"]
        self.redactions.append(self.token)

    def request(self, method, path, body=None, query=None):
        url = API + path + ("?" + urlencode(query) if query else "")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
                status = response.status
                log_id = response.headers.get("x-tt-logid", "")
        except HTTPError as error:
            payload = error.read()
            status = error.code
            log_id = error.headers.get("x-tt-logid", "")
        except (URLError, TimeoutError) as error:
            raise RuntimeError("飞书网络请求失败；写操作结果可能未知，请先核对 Wiki 再重试。") from error
        try:
            result = json.loads(payload)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("飞书返回非 JSON 响应，HTTP %s。" % status) from error
        if status >= 400 or result.get("code") != 0:
            message = str(result.get("msg", "未知错误"))
            for secret in self.redactions:
                message = message.replace(secret, "[REDACTED]")
            raise RuntimeError(
                "飞书 HTTP %s / code=%s: %s（log_id=%s）。"
                "请检查应用权限是否已发布，以及目标知识库是否已授权给应用。"
                % (status, result.get("code"), message, log_id)
            )
        return result

    def node(self, token):
        return self.request("GET", "/wiki/v2/spaces/get_node", query={"token": token})["data"]["node"]

    def list_nodes(self, node):
        items, page_token = [], ""
        while True:
            query = {"parent_node_token": node["node_token"], "page_size": 50}
            if page_token:
                query["page_token"] = page_token
            data = self.request("GET", "/wiki/v2/spaces/%s/nodes" % node["space_id"], query=query)["data"]
            items.extend(data.get("items", []))
            if not data.get("has_more"):
                return items
            page_token = data["page_token"]

    def read(self, node):
        if node["obj_type"] != "docx":
            raise ValueError("实验工作台只支持 docx。")
        return self.request("GET", "/docx/v1/documents/%s/raw_content" % node["obj_token"])["data"]["content"]

    def append(self, node, blocks, operation_id):
        """追加块；超过单次 50 块上限时按顺序拆分，各请求使用派生 client_token。"""
        path = "/docx/v1/documents/{0}/blocks/{0}/children".format(node["obj_token"])
        data = None
        for start in range(0, len(blocks), 50):
            token = operation_id if start == 0 else "%s-%d" % (operation_id, start)
            data = self.request("POST", path, {"children": blocks[start:start + 50], "index": -1},
                                {"document_revision_id": -1, "client_token": token})["data"]
        return data

    def replace(self, node, blocks, operation_id):
        """Append the replacement first, then remove the previous root children."""
        document = node["obj_token"]
        path = "/docx/v1/documents/{0}/blocks/{0}".format(document)
        root = self.request("GET", path, query={"document_revision_id": -1})["data"]["block"]
        old_count = len(root.get("children", []))
        self.append(node, blocks, operation_id)
        remaining = old_count
        while remaining:
            count = min(50, remaining)
            self.request(
                "DELETE",
                path + "/children/batch_delete",
                {"start_index": 0, "end_index": count},
                {"document_revision_id": -1},
            )
            remaining -= count
        return {"removed_blocks": old_count, "replacement_blocks": len(blocks)}


def node_token(value):
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".feishu.cn"):
            raise ValueError("请输入飞书 HTTPS Wiki 地址或节点 token。")
        match = re.fullmatch(r"/wiki/([A-Za-z0-9]+)/?", parsed.path)
        if not match:
            raise ValueError("Wiki 地址路径应为 /wiki/<节点 token>。")
        return match.group(1)
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        raise ValueError("节点 token 仅允许字母和数字。")
    return value


def text_blocks(path):
    text = Path(path).read_text(encoding="utf-8")
    return blocks_from_text(text)


def blocks_from_text(text, limit=50):
    if not text.strip():
        raise ValueError("内容文件不能为空。")
    blocks = []
    for line in text.splitlines():
        # 单个文本元素采用保守长度；每行作为普通文本，不解析 Markdown。
        for start in range(0, max(1, len(line)), 1000):
            blocks.append({"block_type": 2, "text": {"elements": [
                {"text_run": {"content": line[start:start + 1000] or " "}}
            ]}})
    if len(blocks) > limit:
        raise ValueError("单次最多 %d 个文本块，请拆分文件后执行。" % limit)
    return blocks


def wiki_url(config, node):
    origin = urlparse(config["wiki_url"])
    return "%s://%s/wiki/%s" % (origin.scheme, origin.netloc, node["node_token"])


REVIEW_SECTIONS = ("研究问题", "背景与设置", "关键对照", "核心结果", "当前结论", "限制", "下一步")


def validate_review(text):
    lines = {line.strip().rstrip("：:") for line in text.splitlines() if line.strip()}
    missing = [section for section in REVIEW_SECTIONS if section not in lines]
    if missing:
        raise ValueError("审核视图缺少固定栏目：" + "、".join(missing))


def archive_text(path, content):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise ValueError("归档文件已存在且内容不同，请使用新的归档路径。")
        return
    target.write_text(content, encoding="utf-8")


def normalized_text(text):
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def experiment_command(args, config):
    """只管理任务与记录；远端文字不作为 shell 命令自动执行。"""
    if args.command == "record":
        raise ValueError("record 已停用；Wiki 只维护审核视图，请使用 review-update。")

    if args.command == "page-update":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.page_id):
            raise ValueError("page-id 须为 1–80 位字母、数字、点、下划线或连字符。")
        content = Path(args.file).read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError("页面内容不能为空。")
        marker = "协议版本：" + args.page_id
        if marker not in {line.strip() for line in content.splitlines()}:
            raise ValueError("页面须包含独立一行版本标识：%s" % marker)
        guide = config.get("experiment_guide_url")
        if not guide or node_token(args.node) != node_token(guide):
            raise ValueError("page-update 仅用于 configs/feishu.json 登记的实验协议页。")
        client = Feishu()
        node = client.node(node_token(args.node))
        existing = client.read(node)
        if marker in {line.strip() for line in existing.splitlines()}:
            if normalized_text(content) not in normalized_text(existing):
                raise ValueError("相同 page-id 已有不同内容，请使用新的页面版本。")
            return {"written": False, "reason": "该页面版本已存在", "url": wiki_url(config, node)}
        archive_text(args.archive, existing)
        result = client.replace(node, blocks_from_text(content, limit=400),
                                str(uuid.uuid5(uuid.NAMESPACE_URL, node["node_token"] + ":page:" + args.page_id)))
        if normalized_text(content) not in normalized_text(client.read(node)):
            raise RuntimeError("协议页替换后回读不一致，请立即核对页面。")
        return {"written": True, "page_id": args.page_id,
                "archive": str(Path(args.archive).resolve()), "replace": result,
                "url": wiki_url(config, node)}

    if args.command == "review-update":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.review_id):
            raise ValueError("review-id 须为 1–80 位字母、数字、点、下划线或连字符。")
        summary = Path(args.file).read_text(encoding="utf-8").strip()
        if not summary:
            raise ValueError("审核视图不能为空。")
        validate_review(summary)
        content = "审核版本：%s\n状态：%s\n%s" % (args.review_id, args.status, summary)
        blocks = blocks_from_text(content, limit=400)
        client = Feishu()
        node = client.node(node_token(args.node))
        workbenches = {v["workbench_token"] for v in config.get("directions", {}).values()}
        if node.get("parent_node_token") not in workbenches and node.get("node_token") not in workbenches:
            raise ValueError("review-update 仅用于实验工作台或其直属实验任务页面。")
        existing = client.read(node)
        marker = "审核版本：" + args.review_id
        if marker in existing.splitlines():
            if normalized_text(content) not in normalized_text(existing):
                raise ValueError("相同 review-id 已有不同内容，请使用新的审核版本。")
            return {"written": False, "reason": "该审核版本已存在", "url": wiki_url(config, node)}
        archive_text(args.archive, existing)
        operation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, node["node_token"] + ":review:" + args.review_id))
        result = client.replace(node, blocks, operation_id)
        refreshed = client.read(node)
        if (normalized_text(content) not in normalized_text(refreshed)
                or any(line.startswith(("记录 ID：", "结束记录：")) for line in refreshed.splitlines())):
            raise RuntimeError("审核视图替换后回读不一致，请立即核对实验页面。")
        return {"written": True, "review_id": args.review_id, "status": args.status,
                "archive": str(Path(args.archive).resolve()), "replace": result, "url": wiki_url(config, node)}

    direction = config.get("directions", {}).get(args.direction)
    if not direction or not direction.get("workbench_token"):
        raise ValueError("该方向尚未配置实验工作台。")
    if args.command == "experiment-create":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.experiment_id):
            raise ValueError("experiment-id 须为 1–80 位字母、数字、点、下划线或连字符。")
        if not args.title.strip():
            raise ValueError("标题不能为空。")
        plan = Path(args.file).read_text(encoding="utf-8").strip()
        if not plan:
            raise ValueError("实验方案不能为空。")
        validate_review(plan)
        content = "实验 ID：%s\n审核版本：%s-initial\n状态：草案\n%s" % (
            args.experiment_id, args.experiment_id, plan)
        blocks = blocks_from_text(content, limit=400)
    client = Feishu()
    parent = client.node(direction["workbench_token"])
    nodes = client.list_nodes(parent)
    if args.command == "workbench":
        return {"direction": direction["label"], "url": wiki_url(config, parent),
                "content": client.read(parent),
                "experiments": [{"title": n["title"], "node_token": n["node_token"], "url": wiki_url(config, n)} for n in nodes]}
    prefix = "[" + args.experiment_id + "] "
    matches = [n for n in nodes if n["title"].startswith(prefix)]
    if matches:
        if len(matches) != 1:
            raise ValueError("实验 ID 对应多个页面，请先核对目录。")
        node = matches[0]
        # 兼容创建成功、正文写入前中断的情况；不改写非空方案。
        existing = client.read(node).strip()
        if existing and existing != node["title"]:
            raise ValueError("实验 ID 已存在，未新建或覆盖：" + wiki_url(config, node))
        if node["title"] != prefix + args.title.strip():
            raise ValueError("已有空白任务标题不同，请核对后继续。")
    else:
        node = client.request("POST", "/wiki/v2/spaces/%s/nodes" % parent["space_id"], {
            "parent_node_token": parent["node_token"], "obj_type": "docx", "node_type": "origin",
            "title": prefix + args.title.strip(),
        })["data"]["node"]
    print("实验页面：" + wiki_url(config, node), file=sys.stderr, flush=True)
    client.append(node, blocks, str(uuid.uuid5(uuid.NAMESPACE_URL, node["node_token"] + ":initial-plan")))
    if "实验 ID：" + args.experiment_id not in client.read(node).splitlines():
        raise RuntimeError("方案写入后未通过回读，请核对实验页面。")
    return {"created": True, "experiment_id": args.experiment_id, "node_token": node["node_token"], "url": wiki_url(config, node)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("configure", help="交互配置 App ID / App Secret")
    bench = commands.add_parser("workbench", help="读取研究方向的实验逻辑与任务列表")
    bench.add_argument("--direction", choices=("semi_offline", "attention", "hitl"), required=True)
    create = commands.add_parser("experiment-create", help="在实验工作台下创建方案页面（不会启动实验）")
    create.add_argument("--direction", choices=("semi_offline", "attention", "hitl"), required=True)
    create.add_argument("--experiment-id", required=True)
    create.add_argument("--title", required=True)
    create.add_argument("--file", required=True, help="UTF-8 实验方案；单次最多 50 文本块")
    record = commands.add_parser("record", help="已停用；实验页改用 review-update")
    record.add_argument("--node", required=True)
    record.add_argument("--record-id", required=True, help="同一次记录重试沿用同一 ID；修订使用新 ID")
    record.add_argument("--status", choices=STATUSES, required=True)
    record.add_argument("--file", required=True)
    review = commands.add_parser("review-update", help="归档旧正文并替换为面向审核的实验视图")
    review.add_argument("--node", required=True)
    review.add_argument("--review-id", required=True, help="内容修订后使用新的审核版本")
    review.add_argument("--status", choices=STATUSES, required=True)
    review.add_argument("--file", required=True)
    review.add_argument("--archive", required=True, help="替换前 Wiki 正文的本地归档路径")
    page = commands.add_parser("page-update", help="归档旧正文并整页替换登记的实验协议页")
    page.add_argument("--node", required=True)
    page.add_argument("--page-id", required=True, help="须与正文中独立一行“协议版本：<id>”一致；修订使用新 ID")
    page.add_argument("--file", required=True)
    page.add_argument("--archive", required=True, help="替换前 Wiki 正文的本地归档路径")
    for name, help_text in (
        ("check", "只读检查应用凭证、节点与文档访问"),
        ("list", "列出直属子节点"),
        ("read", "读取新版文档纯文本"),
        ("create", "创建空白新版文档子节点"),
        ("append", "向新版文档末尾追加 UTF-8 文件的普通文本"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--node", help="Wiki URL 或节点 token；默认使用项目 Wiki")
        if name == "create":
            sub.add_argument("--title", required=True)
        if name == "append":
            sub.add_argument("--file", required=True)
    args = parser.parse_args()
    if args.command == "configure":
        configure()
        return
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.command in ("workbench", "experiment-create", "record", "review-update", "page-update"):
        print(json.dumps(experiment_command(args, config), ensure_ascii=False, indent=2))
        return
    token = node_token(args.node or config["wiki_url"])
    blocks = text_blocks(args.file) if args.command == "append" else None
    if args.command == "create" and not args.title.strip():
        raise ValueError("标题不能为空。")
    client = Feishu()
    node = client.node(token)
    space_path = "/wiki/v2/spaces/%s/nodes" % node["space_id"]
    if args.command == "list":
        items, page_token = [], ""
        while True:
            query = {"parent_node_token": token, "page_size": 50}
            if page_token:
                query["page_token"] = page_token
            data = client.request("GET", space_path, query=query)["data"]
            items.extend(data.get("items", []))
            if not data.get("has_more"):
                break
            page_token = data["page_token"]
        output = items
    elif args.command == "create":
        output = client.request("POST", space_path, {
            "parent_node_token": token, "obj_type": "docx",
            "node_type": "origin", "title": args.title,
        })["data"]["node"]
    else:
        if node["obj_type"] != "docx":
            if args.command == "check":
                print(json.dumps(node, ensure_ascii=False, indent=2))
                print("节点可访问，但不是新版文档；read/append 仅支持 docx。")
                return
            raise ValueError("read/append 仅支持 docx 新版文档。")
        document_path = "/docx/v1/documents/%s" % node["obj_token"]
        if args.command == "append":
            operation_id = str(uuid.uuid4())
            print("追加请求 ID: " + operation_id, file=sys.stderr, flush=True)
            output = client.request("POST", document_path + "/blocks/%s/children" % node["obj_token"],
                                    {"children": blocks, "index": -1},
                                    {"document_revision_id": -1, "client_token": operation_id})["data"]
        else:
            data = client.request("GET", document_path + "/raw_content")["data"]
            if args.command == "read":
                print(data["content"])
                return
            output = {"node": node, "read_access": True,
                      "write_access": "未验证；check 不执行写操作",
                      "required_tenant_scopes": config["required_tenant_scopes"]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
        print("错误: " + str(exc), file=sys.stderr)
        sys.exit(1)
