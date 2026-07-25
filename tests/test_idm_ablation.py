"""IDM 实验包不依赖真实训练时长的快速回归测试。"""

from __future__ import annotations

import unittest

import torch

from Pipeline.Stages.condense import TopologyCalibrator
from Pipeline.ablation_config import (
    ablation_settings,
    condensation_settings,
    load_ablation_config,
)
from Net.Condensation.idm_official import (
    build_idm_convnet6,
    partition_and_expand,
)


class IDMAblationTests(unittest.TestCase):
    def test_official_configuration_is_locked(self):
        config = load_ablation_config()
        settings = condensation_settings(config)
        idm = settings["idm"]
        self.assertEqual(idm["ipc"], 1)
        self.assertEqual(idm["iterations"], 20000)
        self.assertEqual(idm["batch_real"], 128)
        self.assertEqual(idm["batch_train"], 128)
        self.assertEqual(idm["net_num"], 50)
        self.assertEqual(idm["net_generate_interval"], 50)
        self.assertEqual(idm["train_net_num"], 1)
        self.assertEqual(idm["fetch_net_num"], 2)
        memory = settings["memory"]
        self.assertEqual(memory["evaluation_batch"]["idm_convnet6"], 64)
        self.assertEqual(memory["real_train_microbatch"], 32)
        self.assertLessEqual(memory["max_reserved_fraction"], 0.72)
        self.assertEqual(
            set(ablation_settings(config)["methods"]),
            {f"C{index}" for index in range(6)},
        )

    def test_partition_and_expansion(self):
        images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(
            2, 3, 8, 8
        )
        labels = torch.tensor([1, 3])
        expanded, expanded_labels = partition_and_expand(images, labels, 2)
        self.assertEqual(tuple(expanded.shape), (8, 3, 8, 8))
        self.assertEqual(expanded_labels.tolist(), [1, 3] * 4)

    def test_official_convnet_feature_contract(self):
        model = build_idm_convnet6(3, 4, (224, 224))
        output = model.forward_idm(
            torch.rand(1, 3, 224, 224),
            include_topology=True,
        )
        self.assertEqual(tuple(output.logits.shape), (1, 4))
        self.assertEqual(tuple(output.embedding.shape), (1, 128, 3, 3))
        self.assertEqual(
            set(output.spatial), {"shallow", "middle", "deep"}
        )

    def test_topology_calibration_is_bounded_and_freezes(self):
        settings = {
            "calibration_iterations": [1, 2],
            "target_gradient_fraction": 0.25,
            "minimum_lambda": 0.1,
            "maximum_lambda": 5.0,
        }
        calibrator = TopologyCalibrator(settings)
        value = calibrator.observe(
            1,
            torch.ones(4),
            torch.full((4,), 0.01),
        )
        self.assertEqual(value, 5.0)
        calibrator.observe(2, torch.ones(4), torch.ones(4))
        self.assertTrue(calibrator.frozen)
        self.assertGreaterEqual(calibrator.lambda_value, 0.1)
        self.assertLessEqual(calibrator.lambda_value, 5.0)


if __name__ == "__main__":
    unittest.main()
