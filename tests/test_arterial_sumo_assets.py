import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    '300_0.3': 'sumoarterial1x6_300_03.cfg',
    '300_0.6': 'sumoarterial1x6_300_06.cfg',
    '700_0.3': 'sumoarterial1x6_700_03.cfg',
    '700_0.6': 'sumoarterial1x6_700_06.cfg',
}


def test_all_four_sumo_scenes_resolve_to_existing_common_roadnet():
    roadnets = set()
    routes = set()
    for name in SCENARIOS.values():
        config = json.loads((ROOT / 'configs/sim' / name).read_text())
        roadnet = ROOT / config['dir'] / config['roadnetFile']
        route = ROOT / config['dir'] / config['flowFile']
        combined = ROOT / config['dir'] / config['combined_file']
        assert roadnet.is_file() and route.is_file() and combined.is_file()
        roadnets.add(roadnet.resolve())
        routes.add(route.resolve())
    assert len(roadnets) == 1
    assert len(routes) == 4


def test_six_signal_programs_have_identical_eight_green_actions():
    root = ET.parse(
        ROOT / 'data/raw_data/arterial_1x6_sumo/arterial_1x6.net.xml'
    ).getroot()
    programs = root.findall('tlLogic')
    assert [item.attrib['id'] for item in programs] == [
        f'intersection_{index}_1' for index in range(1, 7)]
    signatures = []
    for program in programs:
        phases = [phase.attrib['state'] for phase in program.findall('phase')]
        green = [state for state in phases
                 if 'y' not in state.lower()
                 and any(letter in state for letter in ('g', 'G'))]
        assert len(green) == 8
        signatures.append(green)
    assert all(value == signatures[0] for value in signatures[1:])
