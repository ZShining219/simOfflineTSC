"""Wiki task lifecycle checks; no network, credentials or simulator required."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import feishu_wiki as wiki


class FakeFeishu:
    def __init__(self):
        self.parent = {"node_token": "bench", "space_id": "space", "obj_type": "docx", "obj_token": "docbench"}
        self.task = {"node_token": "task", "parent_node_token": "bench", "obj_type": "docx", "obj_token": "doctask", "title": "[ATT-1] 测试"}
        self.nodes = []
        self.content = {"bench": "实验逻辑", "task": ""}
        self.writes = 0
        self.replaces = 0

    def node(self, token):
        return self.parent if token == "bench" else self.task

    def list_nodes(self, node):
        return self.nodes

    def read(self, node):
        return self.content[node["node_token"]]

    def request(self, *args):
        self.nodes.append(self.task)
        return {"data": {"node": self.task}}

    def append(self, node, blocks, operation_id):
        self.writes += 1
        self.content[node["node_token"]] += "\n" + "\n".join(b["text"]["elements"][0]["text_run"]["content"] for b in blocks)

    def replace(self, node, blocks, operation_id):
        self.replaces += 1
        self.content[node["node_token"]] = "\n".join(
            b["text"]["elements"][0]["text_run"]["content"] for b in blocks)
        return {"removed_blocks": 3, "replacement_blocks": len(blocks)}


class ExperimentWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "note.txt"
        self.path.write_text("\n".join(("研究问题", "问题", "背景与设置", "设置", "关键对照", "对照",
                                        "核心结果", "尚无", "当前结论", "待验证", "限制", "尚未执行", "下一步", "计划")), encoding="utf-8")
        self.client = FakeFeishu()
        self.config = {"wiki_url": "https://test.feishu.cn/wiki/root", "directions": {"attention": {"label": "注意力", "workbench_token": "bench"}}}

    def run_command(self, command, **kwargs):
        args = argparse.Namespace(command=command, direction="attention", file=str(self.path), **kwargs)
        with patch.object(wiki, "Feishu", return_value=self.client), contextlib.redirect_stderr(io.StringIO()):
            return wiki.experiment_command(args, self.config)

    def test_create_refuses_duplicate_nonempty_task(self):
        result = self.run_command("experiment-create", experiment_id="ATT-1", title="测试")
        self.assertEqual(result["node_token"], "task")
        self.assertIn("状态：草案", self.client.content["task"])
        with self.assertRaisesRegex(ValueError, "实验 ID 已存在"):
            self.run_command("experiment-create", experiment_id="ATT-1", title="测试")
        self.assertEqual(self.client.writes, 1)

    def test_resume_empty_created_page(self):
        self.client.nodes.append(self.client.task)
        self.run_command("experiment-create", experiment_id="ATT-1", title="测试")
        self.assertEqual(len(self.client.nodes), 1)
        self.assertEqual(self.client.writes, 1)

    def test_review_update_replaces_page_and_archives_history(self):
        self.client.content["task"] = "旧过程记录"
        self.path.write_text("\n".join(("研究问题", "问题", "背景与设置", "设置", "关键对照", "对照",
                                        "核心结果", "结果", "当前结论", "结论", "限制", "限制项", "下一步", "计划")), encoding="utf-8")
        archive = Path(self.directory.name) / "history.txt"
        kwargs = dict(node="task", review_id="review-v1", status="已完成", archive=str(archive))
        self.assertTrue(self.run_command("review-update", **kwargs)["written"])
        self.assertEqual(archive.read_text(encoding="utf-8"), "旧过程记录")
        self.assertIn("审核版本：review-v1", self.client.content["task"])
        self.assertNotIn("旧过程记录", self.client.content["task"])
        self.assertFalse(self.run_command("review-update", **kwargs)["written"])
        self.assertEqual(self.client.replaces, 1)

    def test_review_update_requires_audit_sections(self):
        self.path.write_text("研究问题\n问题", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "缺少固定栏目"):
            self.run_command("review-update", node="task", review_id="review-v1",
                             status="草案", archive=str(Path(self.directory.name) / "history.txt"))

    def test_review_retry_ignores_blank_line_representation(self):
        content = "\n".join(("研究问题", "问题", "背景与设置", "设置", "关键对照", "对照",
                              "核心结果", "结果", "当前结论", "结论", "限制", "限制项", "下一步", "计划"))
        self.path.write_text(content, encoding="utf-8")
        self.client.content["task"] = "审核版本：review-v1\n状态：已完成\n" + content.replace("\n\n", "\n \n")
        result = self.run_command("review-update", node="task", review_id="review-v1", status="已完成",
                                  archive=str(Path(self.directory.name) / "history.txt"))
        self.assertFalse(result["written"])
        self.assertEqual(self.client.replaces, 0)

    def test_record_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "已停用"):
            self.run_command("record", node="task", record_id="stage-1", status="执行中")

    def test_review_rejects_outside_workbench(self):
        self.client.task["parent_node_token"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "直属实验"):
            self.path.write_text("\n".join(("研究问题", "问题", "背景与设置", "设置", "关键对照", "对照",
                                            "核心结果", "结果", "当前结论", "结论", "限制", "限制项", "下一步", "计划")), encoding="utf-8")
            self.run_command("review-update", node="task", review_id="r1", status="执行中",
                             archive=str(Path(self.directory.name) / "history.txt"))
        self.assertEqual(self.client.replaces, 0)

    def test_review_accepts_workbench(self):
        self.path.write_text("\n".join(("研究问题", "问题", "背景与设置", "设置", "关键对照", "对照",
                                        "核心结果", "结果", "当前结论", "结论", "限制", "限制项", "下一步", "计划")), encoding="utf-8")
        result = self.run_command("review-update", node="bench", review_id="overview-v1", status="执行中",
                                  archive=str(Path(self.directory.name) / "history.txt"))
        self.assertTrue(result["written"])
        self.assertIn("审核版本：overview-v1", self.client.content["bench"])

    def test_large_plan_rejected_before_network(self):
        self.path.write_text(self.path.read_text(encoding="utf-8") + "\n" + "many\n" * 51)
        with patch.object(wiki, "Feishu", side_effect=AssertionError("must not authenticate")):
            args = argparse.Namespace(command="experiment-create", direction="attention", file=str(self.path), experiment_id="ATT-1", title="测试")
            with self.assertRaisesRegex(ValueError, "50"):
                wiki.experiment_command(args, self.config)

    def test_workbench_has_clickable_task_urls(self):
        self.client.nodes.append(self.client.task)
        result = self.run_command("workbench")
        self.assertEqual(result["experiments"][0]["url"], "https://test.feishu.cn/wiki/task")
        self.assertEqual(result["content"], "实验逻辑")


if __name__ == "__main__":
    unittest.main()
