"""Public workflow contract checks; no ComfyUI or model execution is implied."""
import ast
import copy
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "examples" / "Qwen2.1_精准文案_工作流.json"
API = ROOT / "examples" / "Qwen2.1_精准文案_API.json"


def adapter_contracts():
    """Read the real INPUT_TYPES without importing GPU or Comfy dependencies."""
    source = ast.parse((ROOT / "promo_typesetter" / "nodes.py").read_text())
    styles = next(ast.literal_eval(node.value) for node in source.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "STYLES"
                          for target in node.targets))
    namespace = {"STYLES": styles}
    for node in source.body:
        if isinstance(node, ast.ClassDef) and node.name in ("PromoTextPrepare", "PromoRenderAndSave"):
            method = next(method for method in node.body
                          if isinstance(method, ast.FunctionDef) and method.name == "INPUT_TYPES")
            declaration = ast.ClassDef(name=node.name, bases=[], keywords=[], body=[method], decorator_list=[])
            module = ast.Module(body=[declaration], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), "adapter_INPUT_TYPES", "exec"), namespace)
    return {name: namespace[name].INPUT_TYPES() for name in ("PromoTextPrepare", "PromoRenderAndSave")}


class WorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text())
        self.api = json.loads(API.read_text())
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.links = {link[0]: link for link in self.workflow["links"]}

    def by_type(self, kind):
        matches = [node for node in self.nodes.values() if node["type"] == kind]
        self.assertEqual(len(matches), 1, kind)
        return matches[0]

    def source(self, node, name):
        socket = next(item for item in node["inputs"] if item["name"] == name)
        link = self.links[socket["link"]]
        return link[1], link[2]

    def test_all_links_round_trip_through_source_and_target_sockets(self):
        self.assertEqual(len(self.nodes), len(self.workflow["nodes"]))
        self.assertEqual(len(self.links), len(self.workflow["links"]))
        for link_id, source, source_slot, target, target_slot, kind in self.links.values():
            out_socket = self.nodes[source]["outputs"][source_slot]
            in_socket = self.nodes[target]["inputs"][target_slot]
            self.assertEqual(out_socket["links"].count(link_id), 1)
            self.assertEqual(in_socket["link"], link_id)
            self.assertEqual(out_socket["type"], kind)
            self.assertIn(in_socket["type"], (kind, "*"))
        for node in self.nodes.values():
            for slot, socket in enumerate(node.get("outputs", [])):
                for link_id in socket.get("links", []):
                    self.assertEqual(self.links[link_id][1:3], [node["id"], slot])
            for slot, socket in enumerate(node.get("inputs", [])):
                if socket.get("link") is not None:
                    self.assertEqual(self.links[socket["link"]][3:5], [node["id"], slot])
        self.assertEqual(self.workflow["last_node_id"], max(self.nodes))
        self.assertEqual(self.workflow["last_link_id"], max(self.links))

    def test_native_latent_has_one_sampler_and_one_count_source(self):
        prepare = self.by_type("PromoTextPrepare")
        qwen = self.by_type("TextEncodeQwenImage21")
        repeat = self.by_type("RepeatLatentBatch")
        sampler = self.by_type("KSampler")
        decode = self.by_type("VAEDecode")
        render = self.by_type("PromoRenderAndSave")
        self.assertEqual(self.source(qwen, "images.image_1"), (prepare["id"], 0))
        self.assertEqual(self.source(qwen, "prompt"), (prepare["id"], 2))
        self.assertEqual(self.source(repeat, "samples"), (qwen["id"], 2))
        self.assertEqual(self.source(repeat, "amount"), (prepare["id"], 3))
        self.assertEqual(self.source(sampler, "latent_image"), (repeat["id"], 0))
        self.assertEqual(self.source(decode, "samples"), (sampler["id"], 0))
        self.assertEqual(self.source(render, "images"), (decode["id"], 0))
        self.assertEqual(self.source(render, "context"), (prepare["id"], 1))
        self.assertFalse(any("EmptyLatent" in node["type"] for node in self.nodes.values()))
        self.assertEqual(repeat["widgets_values"], [1])  # Hidden converted widget placeholder.

    def test_required_material_is_connected_to_both_validation_and_qwen(self):
        load = self.by_type("LoadImage")
        self.assertEqual(load["widgets_values"][0], "请选择宣传素材.png")
        self.assertEqual(len(load["outputs"][0]["links"]), 2)
        for kind, name in (("PromoTextPrepare", "material"), ("TextEncodeQwenImage21", "images.image_2")):
            self.assertEqual(self.source(self.by_type(kind), name), (load["id"], 0))

    def test_separate_copy_input_and_adapter_widgets_follow_real_input_contract(self):
        copy_node = self.by_type("PrimitiveStringMultiline")
        prepare = self.by_type("PromoTextPrepare")
        self.assertEqual(copy_node["widgets_values"], ["咖啡还是我的好"])
        self.assertEqual(self.source(prepare, "copy_text"), (copy_node["id"], 0))
        for kind, contract in adapter_contracts().items():
            node = self.by_type(kind)
            # Converted widgets retain their serialized placeholder in ComfyUI.
            descriptors = {**contract["required"], **contract.get("optional", {})}
            widget_names = [name for name, descriptor in descriptors.items()
                            if descriptor[0] not in ("IMAGE", "MASK", "PROMO_CONTEXT")]
            self.assertEqual(len(widget_names), len(node["widgets_values"]))
            api_inputs = self.api[str(node["id"])]["inputs"]
            for name, value in zip(widget_names, node["widgets_values"]):
                if isinstance(api_inputs[name], list):
                    self.assertEqual(value, "", name)
                else:
                    self.assertEqual(value, api_inputs[name], name)
        contract = adapter_contracts()["PromoTextPrepare"]
        self.assertEqual(contract["required"]["style"][0],
                         ["自动", "古风典雅", "现代简约", "时尚潮流", "悬疑暗调", "科技未来"])
        self.assertEqual(self.api[str(prepare["id"])]["inputs"]["width"], 1152)
        self.assertEqual(self.api[str(prepare["id"])]["inputs"]["height"], 1536)

    def test_default_policy_skips_ocr_and_returns_images(self):
        render = self.by_type("PromoRenderAndSave")
        self.assertEqual(render["widgets_values"],
                         ["PromoTypesetter", "off", "return_with_warnings"])
        self.assertEqual(self.api[str(render["id"])]["inputs"]["ocr_mode"], "off")
        required = adapter_contracts()["PromoRenderAndSave"]["required"]
        self.assertEqual(required["ocr_mode"][0], ["off", "reject", "repair_masked"])
        self.assertEqual(self.api[str(render["id"])]["inputs"]["quality_policy"],
                         "return_with_warnings")
        optional = adapter_contracts()["PromoRenderAndSave"]["optional"]
        self.assertEqual(optional["quality_policy"][0], ["return_with_warnings", "strict"])
        self.assertNotIn("验收成品", render["title"])
        note = self.by_type("Note")["widgets_values"][0]
        for term in ("ocr_mode=off", "不运行 OCR", "不需要安装 OCR 环境", "留白", "纹理", "对比度",
                     "不再导致0张输出", "review_required", "warnings", "strict"):
            self.assertIn(term, note)

    def test_api_connections_match_every_gui_link(self):
        for key, node in self.api.items():
            self.assertEqual(node["class_type"], self.nodes[int(key)]["type"])
            for name, value in node["inputs"].items():
                if isinstance(value, list):
                    self.assertEqual(len(value), 2)
                    self.assertIn(value[0], self.api)
                    self.assertEqual(self.source(self.nodes[int(key)], name), (int(value[0]), value[1]))
        for _, source, source_slot, target, target_slot, _ in self.links.values():
            input_name = self.nodes[target]["inputs"][target_slot]["name"]
            self.assertEqual(self.api[str(target)]["inputs"][input_name], [str(source), source_slot])
        qwen = self.api[str(self.by_type("TextEncodeQwenImage21")["id"])]["inputs"]
        self.assertIn("images.image_1", qwen)
        self.assertIn("images.image_2", qwen)
        self.assertEqual(qwen["resolution"], 0)

    def test_requests_for_one_two_three_four_share_one_batch_sampler(self):
        prepare_id = str(self.by_type("PromoTextPrepare")["id"])
        repeat_id = str(self.by_type("RepeatLatentBatch")["id"])
        for count in (1, 2, 3, 4):
            request = copy.deepcopy(self.api)
            request[prepare_id]["inputs"]["count"] = count
            amount_source, amount_slot = request[repeat_id]["inputs"]["amount"]
            self.assertEqual((amount_source, amount_slot), (prepare_id, 3))
            self.assertEqual(request[amount_source]["inputs"]["count"], count)
            self.assertEqual(sum(node["class_type"] == "KSampler" for node in request.values()), 1)

    def test_debug_preview_and_result_status_are_separate_from_saved_outputs(self):
        decode = self.by_type("VAEDecode")
        render = self.by_type("PromoRenderAndSave")
        raw_preview = self.by_type("PreviewImage")
        status = self.by_type("PreviewAny")
        self.assertEqual(self.source(raw_preview, "images"), (decode["id"], 0))
        self.assertIn("未校验底图", raw_preview["title"])
        self.assertIn("不是验收成品", raw_preview["title"])
        self.assertEqual(self.source(status, "source"), (render["id"], 0))
        self.assertEqual(status["inputs"][0]["type"], "*")
        self.assertIn("逐图状态", status["title"])
        self.assertIn("失败原因", status["title"])
        self.assertEqual(raw_preview["widgets_values"], [])
        self.assertEqual(status["widgets_values"], [])
        self.assertEqual(render["outputs"][0]["name"], "result_json")
        self.assertEqual(self.api[str(raw_preview["id"])]["inputs"]["images"], [str(decode["id"]), 0])
        self.assertEqual(self.api[str(status["id"])]["inputs"]["source"], [str(render["id"]), 0])
        # Both display branches preserve the final typesetting and save path.
        self.assertEqual(self.source(render, "images"), (decode["id"], 0))
        self.assertFalse(any(node["type"] == "SaveImage" for node in self.nodes.values()))

    def test_public_template_contains_no_old_plugins_or_private_inputs(self):
        allowed = {"UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "PrimitiveStringMultiline",
                   "PromoTextPrepare", "TextEncodeQwenImage21", "RepeatLatentBatch", "KSampler",
                   "VAEDecode", "PromoRenderAndSave", "PreviewImage", "PreviewAny", "Note"}
        self.assertEqual({node["type"] for node in self.nodes.values()}, allowed)
        serialized = WORKFLOW.read_text() + API.read_text()
        for banned in ("ExactPromoTextPrepare", "DrawText+", "ImageCompositeMasked", '"SaveImage"',
                       "ailink.casa", "/Users/", "/app/", "Screenshot 2026", "nsfw", "nude", "色情"):
            self.assertNotIn(banned, serialized)
        self.assertEqual(self.workflow["extra"]["promo_typesetter_version"], "0.1.3")
        note = self.by_type("Note")["widgets_values"][0]
        self.assertIn("python scripts/setup_ocr.py", note)
        self.assertIn("实际图像仍需逐图验收", note)
        self.assertIn("Prompt executed", note)
        self.assertIn("不代表图片已验收成功", note)
        self.assertIn("README", note)


if __name__ == "__main__":
    unittest.main()
