import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from tools.xiasha_sumo import converter
from tools.xiasha_sumo import evidence


class XiashaSignalAlignmentTest(unittest.TestCase):
    def test_duplicate_source_ids_remain_independent_events(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.csv"
            source.write_text(
                "vehicle_id,semantic_id,entry_edge,exit_edge,entry_time,exit_time\n"
                "same,same,N2J,J2S,1,2\n"
                "same,same,N2J,J2S,3,4\n",
                encoding="utf-8",
            )
            _, valid, *_ = converter.parse_rows(source)

        self.assertEqual(
            [row["_event_id"] for row in valid],
            ["event_000002", "event_000003"],
        )

    def test_logic_line_detectors_cover_all_directed_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "logic.add.xml"
            report = converter.write_logic_line_detectors(
                converter.DEFAULT_NET, output
            )
            detectors = list(ET.parse(output).getroot())
            lanes = {
                detector.get("lane").rsplit("_", 1)[0]
                for detector in detectors
            }
            lane_lengths = {
                lane.get("id"): float(lane.get("length"))
                for edge in ET.parse(converter.DEFAULT_NET).getroot().findall("edge")
                for lane in edge.findall("lane")
            }

        self.assertEqual(
            lanes,
            {"N2J", "S2J", "E2J", "W2J", "J2N", "J2S", "J2E", "J2W"},
        )
        self.assertEqual(report["detector_count"], 18)
        for detector in detectors:
            lane_id = detector.get("lane")
            edge_id = lane_id.rsplit("_", 1)[0]
            expected = converter.logic_line_distance(edge_id)
            if edge_id in converter.CONTROLLED_ENTRY_EDGES:
                expected = lane_lengths[lane_id] - expected
            self.assertAlmostEqual(float(detector.get("pos")), expected, places=2)

    def test_entry_time_is_crossing_time_not_departure_time(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.csv"
            source.write_text(
                "vehicle_id,semantic_id,entry_edge,exit_edge,entry_time,exit_time\n"
                "n,n,N2J,J2S,10,20\n"
                "s,s,S2J,J2N,11,21\n"
                "e,e,E2J,J2W,12,22\n"
                "w,w,W2J,J2E,13,23\n",
                encoding="utf-8",
            )
            _, valid, shift, *_ = converter.parse_rows(source)

        for row in valid:
            pre_roll = converter.logic_line_pre_roll_seconds(row["entry_edge"])
            self.assertAlmostEqual(
                row["_depart"] + pre_roll,
                row["_entry_time"] + shift,
            )
            self.assertGreater(pre_roll, 0.0)

    def test_entry_and_exit_timestamps_compare_at_matching_lines(self):
        frame = pd.DataFrame(
            [
                {
                    "route": "N2J->J2S",
                    "entry_time": 10.0,
                    "exit_time": 20.0,
                    "sumo_logic_entry": 15.2,
                    "sumo_logic_exit": 27.0,
                    "sumo_junction_exit": 24.0,
                }
            ]
        )
        aligned = evidence._attach_logic_line_alignment(frame, shift=5.0).iloc[0]

        self.assertAlmostEqual(aligned["observed_entry_line"], 15.0)
        self.assertAlmostEqual(aligned["entry_line_residual_seconds"], 0.2)
        self.assertAlmostEqual(aligned["observed_exit_line"], 25.0)
        self.assertAlmostEqual(aligned["exit_line_residual_seconds"], 2.0)
        self.assertAlmostEqual(aligned["observed_stopline_proxy"], 22.0)

    def test_od_rank_alignment_keeps_alternating_events_for_validation(self):
        frame = pd.DataFrame(
            {
                "event_id": [f"event_{index:06d}" for index in range(5)],
                "route": ["N2J->J2S"] * 5,
                "turning_direction": ["through"] * 5,
                "observed_exit_line": [30.0, 10.0, 40.0, 0.0, 20.0],
                "sumo_exit_line": [25.0, 45.0, 5.0, 35.0, 15.0],
                "route_post_stopline_seconds": [3.0] * 5,
            }
        )

        aligned, audit = evidence._apply_od_rank_event_alignment(frame)
        ordered = aligned.sort_values("observed_exit_line")

        self.assertEqual(
            ordered["observed_alignment_partition"].tolist(),
            ["calibration", "validation", "calibration", "validation", "calibration"],
        )
        self.assertEqual(
            ordered["event_matched_exit_line"].tolist(),
            [5.0, 15.0, 25.0, 35.0, 45.0],
        )
        self.assertEqual(
            ordered["event_matched_stopline_proxy"].tolist(),
            [2.0, 12.0, 22.0, 32.0, 42.0],
        )
        self.assertEqual(
            [row["partition"] for row in audit],
            ["calibration", "validation", "calibration", "validation", "calibration"],
        )

    def test_phase_position_uses_sumo_subtractive_offset(self):
        # With offset 75.5 s, t=75.5 s is the start of the configured cycle.
        self.assertAlmostEqual(
            converter.signal_phase_position(75.5, 75.5), 0.0
        )
        self.assertEqual(
            converter.signal_phase_at(75.5, 75.5), "NS直行绿"
        )
        # Adding the offset would incorrectly place this timestamp in EW left.
        self.assertNotEqual(
            converter.signal_phase_at(75.5 + 75.5, 75.5), "NS直行绿"
        )

    def test_phase_position_wraps_at_cycle_boundary(self):
        self.assertAlmostEqual(
            converter.signal_phase_position(203.5, 75.5), 0.0
        )
        self.assertAlmostEqual(
            converter.signal_phase_position(-52.5, 75.5), 0.0
        )

    def test_sample_offset_is_separate_from_configured_offset(self):
        # Edge-exit timestamps are interval boundaries; midpoint sampling is
        # an observation convention and must not change the signal offset.
        self.assertAlmostEqual(
            converter.signal_phase_position(
                75.5, 75.5, sample_offset=0.5
            ),
            0.5,
        )
        self.assertAlmostEqual(
            converter.signal_phase_position(75.5, 75.5), 0.0
        )

    def test_parallel_straight_lanes_use_available_outbound_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rebuilt.net.xml"
            report = converter.rebuild_parallel_straight_outbound_lanes(
                converter.DEFAULT_NET, output
            )
            root = ET.parse(output).getroot()

        pairs = {
            (connection.get("from"), connection.get("to")): set()
            for connection in root.findall("connection")
        }
        for connection in root.findall("connection"):
            key = (connection.get("from"), connection.get("to"))
            if key in {("N2J", "J2S"), ("S2J", "J2N")}:
                pairs[key].add(connection.get("toLane"))
        self.assertEqual(report["change_count"], 2)
        self.assertEqual(pairs[("N2J", "J2S")], {"0", "1"})
        self.assertEqual(pairs[("S2J", "J2N")], {"0", "1"})


if __name__ == "__main__":
    unittest.main()
