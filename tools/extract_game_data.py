"""Celeste game data extractor v2 — string-based binary parsing.

Extracts level names, entity types, positions, and tile data from .bin map files
using length-prefixed string scanning, combined with save data statistics.
"""

from __future__ import annotations

import os
import json
import struct
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any


class CelesteBinaryScanner:
    """Scans Celeste .bin files for structured data using string extraction."""

    ENTITY_TYPES = {
        'spikesUp': 'spike_up', 'spikesDown': 'spike_down',
        'spikesLeft': 'spike_left', 'spikesRight': 'spike_right',
        'spinner': 'crystal_spinner', 'crystalSpinner': 'crystal_spinner',
        'spring': 'spring', 'springRight': 'spring_right',
        'springLeft': 'spring_left', 'springUp': 'spring_up',
        'springDown': 'spring_down',
        'dashBlock': 'dash_block', 'dashSwitch': 'dash_switch',
        'booster': 'booster',
        'crumbleBlock': 'crumble_block',
        'swapBlock': 'swap_block',
        'touchSwitch': 'touch_switch',
        'moveBlock': 'move_block',
        'refill': 'refill',
        'strawberry': 'strawberry',
        'wingedStrawberry': 'winged_strawberry',
        'goldenStrawberry': 'golden_strawberry',
        'heart': 'heart_gem',
        'checkpoint': 'checkpoint',
        'cassette': 'cassette',
        'jumpThru': 'jump_thru',
        'zipMover': 'zip_mover',
        'bird': 'bird',
        'exit': 'exit',
        'player_spawn': 'spawn',
        'spawn': 'spawn',
        'ambient': 'ambient',
        'music': 'music',
        'bigSpinner': 'big_spinner',
        'torch': 'torch',
        'cobweb': 'cobweb',
        'banner': 'banner',
        'clutter': 'clutter',
        'clutterBlock': 'clutter_block',
        'core': 'core_mode',
        'floaty': 'floaty',
        'floatySpace': 'floaty_space',
        'floatyTile': 'floaty_tile',
        'intseq': 'interactive_sequence',
        'lightning': 'lightning',
        'trigger': 'trigger',
        'eventTrigger': 'event_trigger',
        'changeRespawnTrigger': 'change_respawn_trigger',
        'checkpointBlocker': 'checkpoint_blocker',
        'everestEventTrigger': 'everest_event_trigger',
        'miniTextbox': 'mini_textbox',
        'textbox': 'textbox',
        'intro': 'intro',
        'wire': 'wire',
        'debris': 'debris',
        'bumpblock': 'bump_block',
        'credits': 'credits',
        'windy': 'windy',
        'summit': 'summit',
        'snow': 'snow',
        'rain': 'rain',
        'parallax': 'parallax',
        'fgParticles': 'fg_particles',
        'dream': 'dream',
        'switchGate': 'switch_gate',
        'refillHeart': 'refill_heart',
        'fakeHeart': 'fake_heart',
        'moon': 'moon',
        'starfield': 'starfield',
        'wavedashtutorial': 'wave_dash_tutorial',
        'hyperdashTutorial': 'hyperdash_tutorial',
        'latentspace': 'latent_space',
        'pico': 'pico',
        'pico8': 'pico8',
        'retry': 'retry',
        'camera': 'camera',
        'cameraOffsetY': 'camera_offset_y',
        'cameraOffsetX': 'camera_offset_x',
    }

    DIFFICULTY_ENTITY_WEIGHTS = {
        'spike': 3, 'crystal_spinner': 4, 'big_spinner': 5,
        'dash_block': 2, 'crumble_block': 2, 'swap_block': 3,
        'move_block': 2, 'zip_mover': 2, 'booster': 2,
        'touch_switch': 2, 'lightning': 4, 'wave_dash_tutorial': 5,
        'hyperdash_tutorial': 5,
    }

    def __init__(self):
        self.stats = defaultdict(int)

    def scan_map(self, filepath: str) -> dict[str, Any]:
        with open(filepath, 'rb') as f:
            data = f.read()

        strings = self._extract_all_strings(data)
        level_names = sorted(set(s for s in strings if re.match(r'^lvl_\w+$', s)))
        entity_raw = [s for s in strings if s in self.ENTITY_TYPES]
        entity_types = [self.ENTITY_TYPES[s] for s in entity_raw]

        entity_counts: dict[str, int] = defaultdict(int)
        for et in entity_types:
            entity_counts[et] += 1

        coord_strings = [s for s in strings if re.match(r'^-?\d+(,-?\d+)+$', s)]

        number_strings = [s for s in strings if re.match(r'^-?\d+$', s) and len(s) <= 6]
        numbers = [int(s) for s in number_strings]

        tile_patterns = [s for s in strings if re.match(r'^-1(,-1)+$', s)]

        music_events = [s for s in strings if s.startswith('event:/') or 'music' in s.lower()]
        graphic_paths = [s for s in strings if '\\' in s and s.endswith('.png')]
        wind_patterns = [s for s in strings if 'wind' in s.lower() or s in ['None', 'Left', 'Right', 'Up', 'Down']]

        has_cassette = 'cassette' in entity_types
        has_heart = 'heart_gem' in entity_types
        has_checkpoint = 'checkpoint' in entity_types

        hazard_count = sum(entity_counts.get(t, 0) for t in
                          ['spike_up', 'spike_down', 'spike_left', 'spike_right',
                           'crystal_spinner', 'big_spinner', 'lightning'])
        mechanic_count = sum(entity_counts.get(t, 0) for t in
                            ['dash_block', 'crumble_block', 'swap_block', 'move_block',
                             'zip_mover', 'booster', 'touch_switch', 'spring',
                             'spring_right', 'spring_left', 'spring_up', 'spring_down'])
        collectible_count = sum(entity_counts.get(t, 0) for t in
                                ['strawberry', 'winged_strawberry', 'golden_strawberry',
                                 'heart_gem', 'cassette'])

        total_entities = sum(entity_counts.values())
        difficulty = self._estimate_difficulty(hazard_count, mechanic_count, collectible_count, len(level_names))

        situations = self._classify_situations(entity_counts, level_names)

        map_name = os.path.basename(filepath).replace('.bin', '')
        area_id = self._extract_area_id(map_name)

        return {
            "map_name": map_name,
            "area_id": area_id,
            "file_size": len(data),
            "level_count": len(level_names),
            "level_names": level_names,
            "total_entities": total_entities,
            "entity_type_counts": dict(entity_counts),
            "hazard_count": hazard_count,
            "mechanic_count": mechanic_count,
            "collectible_count": collectible_count,
            "difficulty": difficulty,
            "situations": situations,
            "has_cassette": has_cassette,
            "has_heart_gem": has_heart,
            "has_checkpoints": has_checkpoint,
            "coord_samples": coord_strings[:10],
            "number_count": len(numbers),
            "number_range": [min(numbers), max(numbers)] if numbers else [0, 0],
            "tile_pattern_count": len(tile_patterns),
            "music_events": list(set(music_events))[:5],
            "graphic_count": len(graphic_paths),
        }

    def _extract_all_strings(self, data: bytes) -> list[str]:
        strings = []
        i = 0
        while i < len(data):
            if data[i] > 1 and i + 1 + data[i] <= len(data):
                length = data[i]
                try:
                    s = data[i + 1:i + 1 + length].decode('utf-8')
                    if all(0x20 <= ord(c) <= 0x7e or c in '\n\r\t' for c in s):
                        strings.append(s)
                        i += 1 + length
                        continue
                except:
                    pass
            i += 1
        return strings

    def _estimate_difficulty(self, hazards: int, mechanics: int, collectibles: int, levels: int) -> str:
        score = hazards * 3 + mechanics * 2 + collectibles + levels
        if score < 20:
            return "easy"
        elif score < 50:
            return "medium"
        elif score < 100:
            return "hard"
        else:
            return "extreme"

    def _classify_situations(self, entity_counts: dict, level_names: list) -> list[str]:
        situations = set()

        if entity_counts.get('spike_up', 0) + entity_counts.get('spike_down', 0) + \
           entity_counts.get('spike_left', 0) + entity_counts.get('spike_right', 0) > 3:
            situations.add("spike_corridor")
        if entity_counts.get('crystal_spinner', 0) + entity_counts.get('big_spinner', 0) > 0:
            situations.add("spinner_field")
        if entity_counts.get('dash_block', 0) > 0:
            situations.add("dash_across_gap")
        if entity_counts.get('booster', 0) > 0:
            situations.add("booster_section")
        if entity_counts.get('spring', 0) + entity_counts.get('spring_up', 0) + \
           entity_counts.get('spring_left', 0) + entity_counts.get('spring_right', 0) > 0:
            situations.add("spring_bounce")
        if entity_counts.get('crumble_block', 0) > 0:
            situations.add("crumble_platform")
        if entity_counts.get('swap_block', 0) > 0:
            situations.add("swap_block_puzzle")
        if entity_counts.get('move_block', 0) + entity_counts.get('zip_mover', 0) > 0:
            situations.add("move_block_puzzle")
        if entity_counts.get('touch_switch', 0) > 0:
            situations.add("touch_switch_puzzle")
        if entity_counts.get('lightning', 0) > 0:
            situations.add("lightning_dodge")
        if entity_counts.get('strawberry', 0) + entity_counts.get('winged_strawberry', 0) > 0:
            situations.add("strawberry_hunt")
        if entity_counts.get('heart_gem', 0) > 0:
            situations.add("heart_gem_challenge")
        if entity_counts.get('checkpoint', 0) > 0:
            situations.add("checkpoint_run")

        if not situations:
            situations.add("simple_move")

        return sorted(situations)

    def _extract_area_id(self, map_name: str) -> int:
        match = re.match(r'^(\d+)', map_name)
        return int(match.group(1)) if match else -1


class SaveDataParser:
    """Parses Celeste .celeste save files (XML format)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()

    def parse(self) -> dict[str, Any]:
        result = {
            "version": self._text("Version"),
            "name": self._text("Name"),
            "total_time_ms": int(self._text("Time", "0")),
            "unlocked_areas": int(self._text("UnlockedAreas", "0")),
            "total_deaths": int(self._text("TotalDeaths", "0")),
            "total_strawberries": int(self._text("TotalStrawberries", "0")),
            "total_golden_strawberries": int(self._text("TotalGoldenStrawberries", "0")),
            "total_jumps": int(self._text("TotalJumps", "0")),
            "total_wall_jumps": int(self._text("TotalWallJumps", "0")),
            "total_dashes": int(self._text("TotalDashes", "0")),
            "areas": [],
        }

        areas_elem = self.root.find("Areas")
        if areas_elem is not None:
            for area in areas_elem.findall("AreaStats"):
                area_id = int(area.get("ID", "0"))
                cassette = area.get("Cassette", "false") == "true"

                modes = []
                for mode in area.find("Modes").findall("AreaModeStats"):
                    mode_data = {
                        "strawberries": int(mode.get("TotalStrawberries", "0")),
                        "completed": mode.get("Completed", "false") == "true",
                        "single_run_completed": mode.get("SingleRunCompleted", "false") == "true",
                        "deaths": int(mode.get("Deaths", "0")),
                        "time_played_ms": int(mode.get("TimePlayed", "0")),
                        "best_time_ms": int(mode.get("BestTime", "0")),
                        "best_dashes": int(mode.get("BestDashes", "0")),
                        "best_deaths": int(mode.get("BestDeaths", "0")),
                        "heart_gem": mode.get("HeartGem", "false") == "true",
                    }
                    modes.append(mode_data)

                result["areas"].append({
                    "id": area_id,
                    "cassette": cassette,
                    "modes": modes,
                })

        return result

    def _text(self, tag: str, default: str = "") -> str:
        elem = self.root.find(tag)
        return elem.text if elem is not None and elem.text else default


class TrainingDataGenerator:
    """Combines map scan data and save data into training samples."""

    AREA_NAMES = {
        0: "Prologue", 1: "Forsaken City", 2: "Old Site",
        3: "Celestial Resort", 4: "Golden Ridge", 5: "Mirror Temple",
        6: "Reflection", 7: "Summit", 8: "Epilogue", 9: "Core",
        10: "Farewell",
    }

    def __init__(self, maps_dir: str, saves_dir: str):
        self.maps_dir = maps_dir
        self.saves_dir = saves_dir
        self.scanner = CelesteBinaryScanner()

    def generate(self, output_dir: str) -> dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        saves = self._parse_saves()
        maps = self._scan_maps()

        samples = self._build_training_samples(maps, saves)

        save_summary = self._summarize_saves(saves)
        map_summary = self._summarize_maps(maps)

        entity_dist = defaultdict(int)
        situation_dist = defaultdict(int)
        difficulty_dist = defaultdict(int)
        for m in maps:
            for et, count in m["entity_type_counts"].items():
                entity_dist[et] += count
            for sit in m["situations"]:
                situation_dist[sit] += 1
            difficulty_dist[m["difficulty"]] += 1

        summary = {
            "saves_parsed": len(saves),
            "maps_scanned": len(maps),
            "total_levels": sum(m["level_count"] for m in maps),
            "total_entities": sum(m["total_entities"] for m in maps),
            "training_samples": len(samples),
            "save_summary": save_summary,
            "map_summary": map_summary,
            "entity_distribution": dict(sorted(entity_dist.items(), key=lambda x: -x[1])),
            "situation_distribution": dict(sorted(situation_dist.items(), key=lambda x: -x[1])),
            "difficulty_distribution": dict(difficulty_dist),
            "area_breakdown": self._area_breakdown(maps, saves),
        }

        with open(os.path.join(output_dir, "training_samples.json"), "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(output_dir, "save_data.json"), "w", encoding="utf-8") as f:
            json.dump(saves, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(output_dir, "map_data.json"), "w", encoding="utf-8") as f:
            json.dump(maps, f, indent=2, ensure_ascii=False, default=str)

        return summary

    def _parse_saves(self) -> list[dict[str, Any]]:
        results = []
        for i in range(3):
            path = os.path.join(self.saves_dir, f"{i}.celeste")
            if os.path.exists(path):
                try:
                    parser = SaveDataParser(path)
                    results.append(parser.parse())
                except Exception as e:
                    results.append({"slot": i, "error": str(e)})
        return results

    def _scan_maps(self) -> list[dict[str, Any]]:
        results = []
        if not os.path.isdir(self.maps_dir):
            return results
        for fname in sorted(os.listdir(self.maps_dir)):
            if not fname.endswith(".bin"):
                continue
            filepath = os.path.join(self.maps_dir, fname)
            try:
                data = self.scanner.scan_map(filepath)
                results.append(data)
            except Exception as e:
                results.append({"map_name": fname, "error": str(e)})
        return results

    def _build_training_samples(self, maps: list, saves: list) -> list[dict[str, Any]]:
        samples = []

        for map_info in maps:
            if "error" in map_info:
                continue

            area_id = map_info.get("area_id", -1)
            save_stats = self._find_save_stats(saves, area_id)

            sample = {
                "type": "map_training_sample",
                "map_name": map_info["map_name"],
                "area_id": area_id,
                "area_name": self.AREA_NAMES.get(area_id, "Unknown"),
                "level_count": map_info["level_count"],
                "level_names": map_info["level_names"],
                "total_entities": map_info["total_entities"],
                "entity_types": map_info["entity_type_counts"],
                "hazard_count": map_info["hazard_count"],
                "mechanic_count": map_info["mechanic_count"],
                "collectible_count": map_info["collectible_count"],
                "difficulty": map_info["difficulty"],
                "situations": map_info["situations"],
                "has_cassette": map_info["has_cassette"],
                "has_heart_gem": map_info["has_heart_gem"],
                "player_deaths_a_side": save_stats.get("a_side_deaths", 0),
                "player_deaths_b_side": save_stats.get("b_side_deaths", 0),
                "player_deaths_c_side": save_stats.get("c_side_deaths", 0),
                "player_completed_a": save_stats.get("a_completed", False),
                "player_completed_b": save_stats.get("b_completed", False),
                "player_best_time_a": save_stats.get("a_best_time", 0),
                "player_best_dashes_a": save_stats.get("a_best_dashes", 0),
            }
            samples.append(sample)

            for level_name in map_info["level_names"]:
                level_sample = {
                    "type": "level_training_sample",
                    "map_name": map_info["map_name"],
                    "level_name": level_name,
                    "area_id": area_id,
                    "area_name": self.AREA_NAMES.get(area_id, "Unknown"),
                    "entity_types_in_map": map_info["entity_type_counts"],
                    "map_difficulty": map_info["difficulty"],
                    "map_situations": map_info["situations"],
                    "player_deaths_in_area": save_stats.get("a_side_deaths", 0),
                }
                samples.append(level_sample)

        return samples

    def _find_save_stats(self, saves: list, area_id: int) -> dict[str, Any]:
        for save in saves:
            if "error" in save:
                continue
            for area in save.get("areas", []):
                if area["id"] == area_id:
                    modes = area.get("modes", [])
                    return {
                        "a_side_deaths": modes[0]["deaths"] if len(modes) > 0 else 0,
                        "b_side_deaths": modes[1]["deaths"] if len(modes) > 1 else 0,
                        "c_side_deaths": modes[2]["deaths"] if len(modes) > 2 else 0,
                        "a_completed": modes[0]["completed"] if len(modes) > 0 else False,
                        "b_completed": modes[1]["completed"] if len(modes) > 1 else False,
                        "a_best_time": modes[0]["best_time_ms"] if len(modes) > 0 else 0,
                        "a_best_dashes": modes[0]["best_dashes"] if len(modes) > 0 else 0,
                    }
        return {}

    def _summarize_saves(self, saves: list) -> dict[str, Any]:
        if not saves:
            return {}
        return {
            "save_count": len(saves),
            "players": [
                {
                    "name": s.get("name", "?"),
                    "deaths": s.get("total_deaths", 0),
                    "jumps": s.get("total_jumps", 0),
                    "dashes": s.get("total_dashes", 0),
                    "strawberries": s.get("total_strawberries", 0),
                    "unlocked_areas": s.get("unlocked_areas", 0),
                    "total_time_hours": s.get("total_time_ms", 0) / 3.6e9,
                }
                for s in saves if "error" not in s
            ],
            "totals": {
                "deaths": sum(s.get("total_deaths", 0) for s in saves if "error" not in s),
                "jumps": sum(s.get("total_jumps", 0) for s in saves if "error" not in s),
                "dashes": sum(s.get("total_dashes", 0) for s in saves if "error" not in s),
                "strawberries": sum(s.get("total_strawberries", 0) for s in saves if "error" not in s),
            },
        }

    def _summarize_maps(self, list) -> dict[str, Any]:
        valid = [m for m in list if "error" not in m]
        return {
            "map_count": len(valid),
            "total_levels": sum(m["level_count"] for m in valid),
            "total_entities": sum(m["total_entities"] for m in valid),
            "total_hazards": sum(m["hazard_count"] for m in valid),
            "total_mechanics": sum(m["mechanic_count"] for m in valid),
            "total_collectibles": sum(m["collectible_count"] for m in valid),
            "avg_levels_per_map": sum(m["level_count"] for m in valid) / max(1, len(valid)),
            "avg_entities_per_map": sum(m["total_entities"] for m in valid) / max(1, len(valid)),
        }

    def _area_breakdown(self, maps: list, saves: list) -> list[dict[str, Any]]:
        breakdown = []
        for map_info in sorted(maps, key=lambda m: m.get("area_id", 99)):
            if "error" in map_info:
                continue
            area_id = map_info.get("area_id", -1)
            save_stats = self._find_save_stats(saves, area_id)
            breakdown.append({
                "area_id": area_id,
                "area_name": self.AREA_NAMES.get(area_id, "Unknown"),
                "map_name": map_info["map_name"],
                "levels": map_info["level_count"],
                "entities": map_info["total_entities"],
                "hazards": map_info["hazard_count"],
                "difficulty": map_info["difficulty"],
                "situations": map_info["situations"],
                "player_deaths_a": save_stats.get("a_side_deaths", 0),
                "player_deaths_b": save_stats.get("b_side_deaths", 0),
                "player_completed_a": save_stats.get("a_completed", False),
            })
        return breakdown


if __name__ == "__main__":
    maps_dir = "/Users/zhaoxiangyu/Library/Application Support/Steam/steamapps/common/Celeste/Celeste.app/Contents/Resources/Content/Maps"
    saves_dir = "/Users/zhaoxiangyu/Library/Application Support/Celeste/Saves"
    output_dir = "/Users/zhaoxiangyu/.trae-cn/assistant/celeste-jev-rl/data/training"

    gen = TrainingDataGenerator(maps_dir, saves_dir)
    summary = gen.generate(output_dir)

    print("=" * 60)
    print("  Celeste Game Data Extraction — Complete")
    print("=" * 60)
    print(f"  Maps scanned:      {summary['maps_scanned']}")
    print(f"  Total levels:      {summary['total_levels']}")
    print(f"  Total entities:    {summary['total_entities']}")
    print(f"  Training samples:  {summary['training_samples']}")
    print()
    print("  Save Data Summary:")
    ss = summary["save_summary"]
    print(f"    Saves:           {ss['save_count']}")
    print(f"    Total deaths:    {ss['totals']['deaths']}")
    print(f"    Total jumps:     {ss['totals']['jumps']}")
    print(f"    Total dashes:    {ss['totals']['dashes']}")
    print(f"    Strawberries:    {ss['totals']['strawberries']}")
    for p in ss["players"]:
        print(f"    Player '{p['name']}': {p['deaths']} deaths, {p['strawberries']} strawberries, {p['unlocked_areas']} areas unlocked, {p['total_time_hours']:.1f}h playtime")
    print()
    print("  Map Summary:")
    ms = summary["map_summary"]
    print(f"    Maps:            {ms['map_count']}")
    print(f"    Total levels:    {ms['total_levels']}")
    print(f"    Total entities:  {ms['total_entities']}")
    print(f"    Total hazards:   {ms['total_hazards']}")
    print(f"    Total mechanics: {ms['total_mechanics']}")
    print(f"    Avg levels/map:  {ms['avg_levels_per_map']:.1f}")
    print(f"    Avg entities/map:{ms['avg_entities_per_map']:.1f}")
    print()
    print("  Entity Distribution (top 20):")
    for ent, count in list(summary["entity_distribution"].items())[:20]:
        print(f"    {ent:30s} {count:5d}")
    print()
    print("  Situation Distribution:")
    for sit, count in summary["situation_distribution"].items():
        print(f"    {sit:30s} {count:5d}")
    print()
    print("  Difficulty Distribution:")
    for diff, count in summary["difficulty_distribution"].items():
        print(f"    {diff:15s} {count:5d}")
    print()
    print("  Area Breakdown:")
    for area in summary["area_breakdown"]:
        print(f"    [{area['area_id']:2d}] {area['area_name']:20s} "
              f"levels={area['levels']:3d} entities={area['entities']:4d} "
              f"hazards={area['hazards']:3d} diff={area['difficulty']:8s} "
              f"deaths_a={area['player_deaths_a']:4d} "
              f"done={'Y' if area['player_completed_a'] else 'N'}")
    print()
    print(f"  Output: {output_dir}/")
    print("=" * 60)
