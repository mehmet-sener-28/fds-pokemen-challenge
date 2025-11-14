import numpy as np
import pandas as pd

from .config import COL_TARGET, COL_ID, SEED

# =============================================================================
# CONSTANTS & HELPERLAR
# =============================================================================

HIGHVALUE_SLEEP_TARGETS = {"tauros", "snorlax", "starmie", "alakazam", "jynx"}
HIGHVALUE_PARALYZE_TARGETS = {"tauros", "alakazam", "starmie", "zapdos", "jolteon"}
SPECIAL_WALLS_FOR_FREEZE = {"chansey", "starmie"}
PHYSICAL_TITANS_FOR_BURN = {"tauros", "snorlax", "rhydon", "golem"}

STATUS_CODES = {"slp", "par", "frz", "brn"}
EARLY_TURN_THRESHOLD = 5
LATE_TURN_THRESHOLD = 20

STATUS_SEVERITY = {"slp": 3.0, "frz": 2.5, "par": 1.5, "brn": 1.2}
TEMPORAL_DECAY = 0.95
LATE_GAME_CONTROL_THRESHOLD = 2

EFFECT_WEIGHTS = {
    "reflect": 2.0,
    "substitute": 1.5,
    "leech seed": 1.0,
    "light screen": 0.5,
}

ROLE_TAXONOMY = {
    "wall_phys": {"cloyster", "golem", "rhydon", "articuno"},
    "wall_spec": {"chansey", "snorlax"},
    "breaker_phys": {"tauros", "snorlax"},
    "breaker_spec": {"alakazam", "starmie", "jolteon", "zapdos", "jynx", "exeggutor"},
    "status_spreader": {"jynx", "exeggutor", "chansey", "starmie", "zapdos", "gengar"},
}

SETUP_MOVES = {"amnesia", "swordsdance"}
WALL_MOVES = {"reflect", "rest", "recover"}
STATUS_MOVES = {"thunderwave", "sleeppowder", "lovelykiss", "hypnosis", "stunspore"}

# Pokémon değerleri (HP, kritik vs için)
POKEMON_VALUES = {
    "tauros": 1.5,
    "chansey": 1.2,
    "zapdos": 1.0,
    "exeggutor": 1.0,
    "alakazam": 1.0,
    "slowbro": 1.0,
    "starmie": 0.9,
    "snorlax": 1.2,
    "jynx": 0.8,
    "lapras": 0.8,
}

TYPE_ADVANTAGES = {
    "water": {"fire", "ground", "rock"},
    "fire": {"grass", "ice", "bug"},
    "grass": {"water", "ground", "rock"},
    "electric": {"water", "flying"},
    "ice": {"grass", "ground", "flying", "dragon"},
    "fighting": {"normal", "ice", "rock"},
    "poison": {"grass", "bug"},
    "ground": {"fire", "electric", "poison", "rock"},
    "flying": {"grass", "fighting", "bug"},
    "psychic": {"fighting", "poison"},
    "bug": {"grass", "poison", "psychic"},
    "rock": {"fire", "ice", "flying", "bug"},
    "ghost": {"ghost", "psychic"},
    "dragon": {"dragon"},
}

POKEMON_TYPES = {
    "alakazam": ["psychic"],
    "articuno": ["flying", "ice"],
    "chansey": ["normal"],
    "charizard": ["fire", "flying"],
    "cloyster": ["ice", "water"],
    "dragonite": ["dragon", "flying"],
    "exeggutor": ["grass", "psychic"],
    "gengar": ["ghost", "poison"],
    "golem": ["ground", "rock"],
    "jolteon": ["electric"],
    "jynx": ["ice", "psychic"],
    "lapras": ["ice", "water"],
    "persian": ["normal"],
    "rhydon": ["ground", "rock"],
    "slowbro": ["psychic", "water"],
    "snorlax": ["normal"],
    "starmie": ["psychic", "water"],
    "tauros": ["normal"],
    "victreebel": ["grass", "poison"],
    "zapdos": ["electric", "flying"],
}

SPEED_TIERS = {
    "jolteon": 130,
    "alakazam": 120,
    "persian": 115,
    "starmie": 115,
    "gengar": 110,
    "tauros": 110,
    "charizard": 100,
    "zapdos": 100,
    "jynx": 95,
    "articuno": 85,
    "dragonite": 80,
    "cloyster": 70,
    "victreebel": 70,
    "lapras": 60,
    "exeggutor": 55,
    "chansey": 50,
    "golem": 45,
    "rhydon": 40,
    "snorlax": 30,
    "slowbro": 30,
}

SCORE_WEIGHTS = {
    "highvalue": 3.0,
    "hp": 2.0,
    "status": 1.5,
    "role": 1.0,
    "tempo": 0.5,
}
ERROR_THRESHOLD = -1.0
BLUNDER_THRESHOLD = -2.0


# =============================================================================
# HELPER FONKSİYONLAR
# =============================================================================

def _collect_status_events(row) -> list:
    timeline = row.get("battle_timeline", []) or []
    events = [
        {
            "turn": int(t.get("turn", 0)),
            "side": side,
            "target": ((t.get(f"{side}_pokemon_state", {}) or {}).get("name") or "").lower(),
            "status": (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus"),
        }
        for t in timeline
        for side in ("p1", "p2")
        if (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus") in STATUS_CODES
    ]
    return sorted(events, key=lambda e: e["turn"])


def _is_highvalue_target(pokemon_name: str) -> bool:
    return pokemon_name in (
        HIGHVALUE_SLEEP_TARGETS
        | HIGHVALUE_PARALYZE_TARGETS
        | SPECIAL_WALLS_FOR_FREEZE
        | PHYSICAL_TITANS_FOR_BURN
    )


def _get_pokemon_roles(pokemon_name: str, moves_used: set = None) -> set:
    pokemon_name = pokemon_name.lower()
    roles = {role for role, pokemon_set in ROLE_TAXONOMY.items() if pokemon_name in pokemon_set}

    if moves_used:
        moves_lower = {m.lower() for m in moves_used}
        if moves_lower & SETUP_MOVES:
            roles.add("bulky_setup")
        if moves_lower & WALL_MOVES:
            roles.add("wall_spec" if pokemon_name in {"snorlax", "chansey"} else "wall_phys")
        if moves_lower & STATUS_MOVES:
            roles.add("status_spreader")

    return roles


# ---- Role & endgame helpers ----

def _extract_survivor_roles(row):
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {role: (0, 0) for role in ROLE_TAXONOMY.keys()}

    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return {role: (0, 0) for role in ROLE_TAXONOMY.keys()}

    target_turn = min(30, max(turns))

    p1_fainted = set()
    p2_fainted = set()
    p1_moves = {}
    p2_moves = {}

    for t in timeline:
        turn = int(t.get("turn", 0))
        if turn > target_turn:
            break

        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}

        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()

        if p1_state.get("hp_pct", 1.0) == 0 and p1_name:
            p1_fainted.add(p1_name)
        if p2_state.get("hp_pct", 1.0) == 0 and p2_name:
            p2_fainted.add(p2_name)

        p1_move = t.get("p1_move_details", {}) or {}
        p2_move = t.get("p2_move_details", {}) or {}

        if p1_name:
            p1_moves.setdefault(p1_name, set())
            move_name = (p1_move.get("name") or "").lower()
            if move_name:
                p1_moves[p1_name].add(move_name)

        if p2_name:
            p2_moves.setdefault(p2_name, set())
            move_name = (p2_move.get("name") or "").lower()
            if move_name:
                p2_moves[p2_name].add(move_name)

    p1_team = set()
    p2_team = set()
    for t in timeline[:target_turn]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}

        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()

        if p1_name:
            p1_team.add(p1_name)
        if p2_name:
            p2_team.add(p2_name)

    p1_survivors = p1_team - p1_fainted
    p2_survivors = p2_team - p2_fainted

    role_counts = {}
    for role in ROLE_TAXONOMY.keys():
        p1_count = 0
        p2_count = 0

        for pokemon in p1_survivors:
            moves_used = p1_moves.get(pokemon, set())
            roles = _get_pokemon_roles(pokemon, moves_used)
            if role in roles:
                p1_count += 1

        for pokemon in p2_survivors:
            moves_used = p2_moves.get(pokemon, set())
            roles = _get_pokemon_roles(pokemon, moves_used)
            if role in roles:
                p2_count += 1

        role_counts[role] = (p1_count, p2_count)

    return role_counts


def _count_kos_dealt(row, side):
    timeline = row.get("battle_timeline", []) or []
    opponent_side = "p2" if side == "p1" else "p1"

    fainted = set()
    for t in timeline[:30]:
        state = t.get(f"{opponent_side}_pokemon_state", {}) or {}
        if state.get("hp_pct", 1.0) == 0:
            name = (state.get("name") or "").lower()
            if name:
                fainted.add(name)

    return len(fainted)


def _calculate_hp_distribution_features(row):
    timeline = row.get("battle_timeline", []) or []
    default_result = {
        "p1_avg_hp": 0,
        "p2_avg_hp": 0,
        "p1_std_hp": 0,
        "p2_std_hp": 0,
        "p1_cv_hp": 0,
        "p2_cv_hp": 0,
        "p1_median_hp": 0,
        "p2_median_hp": 0,
        "p1_high_dispersion": 0,
        "p2_high_dispersion": 0,
        "p1_weighted_avg_hp": 0,
        "p2_weighted_avg_hp": 0,
        "p1_effective_avg_hp": 0,
        "p2_effective_avg_hp": 0,
    }

    if not timeline:
        return default_result

    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return default_result

    target_turn = min(30, max(turns))

    p1_pokemon_hp, p2_pokemon_hp = {}, {}
    p1_pokemon_status, p2_pokemon_status = {}, {}
    p1_fainted, p2_fainted = set(), set()

    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break

        for side, hp_dict, status_dict, fainted_set in [
            ("p1", p1_pokemon_hp, p1_pokemon_status, p1_fainted),
            ("p2", p2_pokemon_hp, p2_pokemon_status, p2_fainted),
        ]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            name = (state.get("name") or "").lower()
            if name:
                hp_pct = state.get("hp_pct", 0.0)
                hp_dict[name] = hp_pct
                status_dict[name] = state.get("status", "nostatus")
                if hp_pct == 0:
                    fainted_set.add(name)

    p1_alive_hp = [hp for name, hp in p1_pokemon_hp.items() if name not in p1_fainted and hp > 0]
    p2_alive_hp = [hp for name, hp in p2_pokemon_hp.items() if name not in p2_fainted and hp > 0]

    def calc_stats(alive_hp, pokemon_hp, pokemon_status, fainted):
        if not alive_hp:
            return 0, 0, 0, 0, 0, 0, 0

        avg = np.mean(alive_hp)
        median = np.median(alive_hp)
        std = min(np.std(alive_hp), 0.5) if len(alive_hp) > 1 else 0.0
        cv = (std / avg) if avg > 0 else 0.0
        high_disp = 1 if std > 0.25 else 0

        weighted_hp = [
            hp * POKEMON_VALUES.get(name, 0.6)
            for name, hp in pokemon_hp.items()
            if name not in fainted and hp > 0
        ]
        weighted_avg = np.mean(weighted_hp) if weighted_hp else 0.0

        status_penalty = {"slp": 0.8, "frz": 0.6, "par": 0.75, "brn": 0.6}
        effective_hp = [
            hp * status_penalty.get(pokemon_status.get(name, "nostatus"), 1.0)
            for name, hp in pokemon_hp.items()
            if name not in fainted and hp > 0
        ]
        effective_avg = np.mean(effective_hp) if effective_hp else 0.0

        return avg, median, std, cv, high_disp, weighted_avg, effective_avg

    p1_avg, p1_median, p1_std, p1_cv, p1_high_disp, p1_weighted_avg, p1_effective_avg = calc_stats(
        p1_alive_hp, p1_pokemon_hp, p1_pokemon_status, p1_fainted
    )
    p2_avg, p2_median, p2_std, p2_cv, p2_high_disp, p2_weighted_avg, p2_effective_avg = calc_stats(
        p2_alive_hp, p2_pokemon_hp, p2_pokemon_status, p2_fainted
    )

    return {
        "p1_avg_hp": p1_avg,
        "p2_avg_hp": p2_avg,
        "p1_std_hp": p1_std,
        "p2_std_hp": p2_std,
        "p1_cv_hp": p1_cv,
        "p2_cv_hp": p2_cv,
        "p1_median_hp": p1_median,
        "p2_median_hp": p2_median,
        "p1_high_dispersion": p1_high_disp,
        "p2_high_dispersion": p2_high_disp,
        "p1_weighted_avg_hp": p1_weighted_avg,
        "p2_weighted_avg_hp": p2_weighted_avg,
        "p1_effective_avg_hp": p1_effective_avg,
        "p2_effective_avg_hp": p2_effective_avg,
    }


def _extract_status_at_t30(row):
    timeline = row.get("battle_timeline", []) or []
    default = {
        "p1_statused_alive": 0,
        "p2_statused_alive": 0,
        "p1_sleepers": 0,
        "p2_sleepers": 0,
        "p1_freezes": 0,
        "p2_freezes": 0,
        "p1_paras": 0,
        "p2_paras": 0,
    }

    if not timeline:
        return default

    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return default

    target_turn = min(30, max(turns))

    p1_pokemon_status, p2_pokemon_status = {}, {}
    p1_fainted, p2_fainted = set(), set()

    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break

        for side, status_dict, fainted_set in [
            ("p1", p1_pokemon_status, p1_fainted),
            ("p2", p2_pokemon_status, p2_fainted),
        ]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            name = (state.get("name") or "").lower()
            if name:
                status_dict[name] = state.get("status", "nostatus")
                if state.get("hp_pct", 1.0) == 0:
                    fainted_set.add(name)

    def count_status(pokemon_status, fainted):
        statused = sleepers = freezes = paras = 0
        for pokemon, status in pokemon_status.items():
            if pokemon not in fainted and status in STATUS_CODES:
                statused += 1
                if status == "slp":
                    sleepers += 1
                elif status == "frz":
                    freezes += 1
                elif status == "par":
                    paras += 1
        return statused, sleepers, freezes, paras

    p1_statused, p1_sleepers, p1_freezes, p1_paras = count_status(
        p1_pokemon_status, p1_fainted
    )
    p2_statused, p2_sleepers, p2_freezes, p2_paras = count_status(
        p2_pokemon_status, p2_fainted
    )

    return {
        "p1_statused_alive": p1_statused,
        "p2_statused_alive": p2_statused,
        "p1_sleepers": p1_sleepers,
        "p2_sleepers": p2_sleepers,
        "p1_freezes": p1_freezes,
        "p2_freezes": p2_freezes,
        "p1_paras": p1_paras,
        "p2_paras": p2_paras,
    }


def _first_status(events, side):
    return next((e for e in events if e["side"] == side), None)


def _count_highvalue(events, side, target_pool=None):
    return sum(
        1
        for e in events
        if e["side"] == side and (target_pool is None or e["target"] in target_pool)
    )


def _late_game_flag(events):
    return 1 if any(e["turn"] >= LATE_TURN_THRESHOLD for e in events) else 0


def _turns_disabled_diff(row):
    p1_disabled = p2_disabled = 0
    for t in row.get("battle_timeline", []) or []:
        p1s = (t.get("p1_pokemon_state", {}) or {}).get("status", "nostatus")
        p2s = (t.get("p2_pokemon_state", {}) or {}).get("status", "nostatus")
        p1_disabled += p1s in {"slp", "frz"}
        p2_disabled += p2s in {"slp", "frz"}
    return p1_disabled - p2_disabled


def _turns_disabled_diff_temporal(row) -> tuple:
    counts = {"p1_early": 0, "p2_early": 0, "p1_late": 0, "p2_late": 0}

    for t in row.get("battle_timeline", []) or []:
        turn = int(t.get("turn", 0))
        for side in ["p1", "p2"]:
            status = (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus")
            if status in {"slp", "frz"}:
                if turn <= EARLY_TURN_THRESHOLD:
                    counts[f"{side}_early"] += 1
                elif turn >= LATE_TURN_THRESHOLD:
                    counts[f"{side}_late"] += 1

    return (
        counts["p1_early"] - counts["p2_early"],
        counts["p1_late"] - counts["p2_late"],
    )


def _status_diff_highvalue_w(row) -> float:
    events = _collect_status_events(row)
    scores = {"p1": 0.0, "p2": 0.0}

    for e in events:
        if _is_highvalue_target(e["target"]):
            scores[e["side"]] += STATUS_SEVERITY.get(e["status"], 1.0)

    return scores["p1"] - scores["p2"]


def _late_game_status_swing_v2(row) -> int:
    events = _collect_status_events(row)
    counts = {"early_p1": 0, "early_p2": 0, "late_p1": 0, "late_p2": 0}

    for e in events:
        prefix = (
            "early"
            if e["turn"] <= EARLY_TURN_THRESHOLD
            else "late" if e["turn"] >= LATE_TURN_THRESHOLD else None
        )
        if prefix:
            counts[f"{prefix}_{e['side']}"] += 1

    early_diff = counts["early_p1"] - counts["early_p2"]
    late_diff = counts["late_p1"] - counts["late_p2"]

    return (
        1
        if late_diff > early_diff
        else -1
        if abs(early_diff - late_diff) >= LATE_GAME_CONTROL_THRESHOLD
        else 0
    )


def _turns_disabled_diff_w_decay(row) -> float:
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return 0.0

    max_turn = max(int(t.get("turn", 0)) for t in timeline)
    scores = {"p1": 0.0, "p2": 0.0}

    for t in timeline:
        turn = int(t.get("turn", 0))
        decay_factor = TEMPORAL_DECAY ** (max_turn - turn)

        for side in ["p1", "p2"]:
            status = (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus")
            if status in STATUS_SEVERITY:
                scores[side] += STATUS_SEVERITY[status] * decay_factor

    return scores["p2"] - scores["p1"]


def _extract_endgame_state(row):
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return 0, 0, 0, 0, {}, {}

    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return 0, 0, 0, 0, {}, {}

    target_turn = min(30, max(turns))

    p1_fainted_names, p2_fainted_names = set(), set()

    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break

        for side, fainted_set in [("p1", p1_fainted_names), ("p2", p2_fainted_names)]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            if state.get("hp_pct", 1.0) == 0:
                name = state.get("name", "")
                if name:
                    fainted_set.add(name.lower())

    p1_kos = max(0, min(6, len(p1_fainted_names)))
    p2_kos = max(0, min(6, len(p2_fainted_names)))

    p1_survivors = 6 - p1_kos
    p2_survivors = 6 - p2_kos

    last_state = timeline[min(target_turn - 1, len(timeline) - 1)]

    def extract_effects(state):
        effects = {}
        effects_list = state.get("effects", []) or []
        if isinstance(effects_list, list):
            for effect in effects_list:
                effect_str = str(effect).lower()
                if effect_str == "noeffect":
                    continue
                for key in ["reflect", "substitute", "leech seed", "light screen"]:
                    if key.replace(" ", "") in effect_str.replace(" ", ""):
                        effects[key] = 1
                        break
        return effects

    p1_effects = extract_effects(last_state.get("p1_pokemon_state", {}) or {})
    p2_effects = extract_effects(last_state.get("p2_pokemon_state", {}) or {})

    return p1_kos, p2_kos, p1_survivors, p2_survivors, p1_effects, p2_effects


def _calculate_position_score(state_info: dict) -> float:
    if not state_info["alive_pokemon"]:
        return -10.0

    alive = state_info["alive_pokemon"]

    highvalue_sum = sum(
        POKEMON_VALUES.get(name.lower(), 0.6) * hp_pct for name, hp_pct, _ in alive
    )
    avg_hp = np.mean([hp for _, hp, _ in alive])
    status_burden = -sum(1 for _, _, status in alive if status in STATUS_CODES) / len(alive)
    role_score = min(
        sum(count * 0.3 for count in state_info["roles"].values()) / 3.0, 1.0
    )
    tempo = SCORE_WEIGHTS["tempo"] if state_info.get("opponent_disabled", False) else 0

    return (
        SCORE_WEIGHTS["highvalue"] * (highvalue_sum / 6.0)
        + SCORE_WEIGHTS["hp"] * avg_hp
        + SCORE_WEIGHTS["status"] * status_burden
        + SCORE_WEIGHTS["role"] * role_score
        + tempo
    )


def _get_state_snapshot(timeline, turn_index, side):
    if turn_index >= len(timeline):
        return None

    pokemon_last_state = {}
    fainted = set()

    for t in timeline[: turn_index + 1]:
        st = t.get(f"{side}_pokemon_state", {}) or {}
        name = (st.get("name") or "").lower()
        if name:
            hp_pct = st.get("hp_pct", 1.0)
            if hp_pct == 0:
                fainted.add(name)
            else:
                pokemon_last_state[name] = (hp_pct, st.get("status", "nostatus"))

    alive_pokemon = [
        (name, hp, status)
        for name, (hp, status) in pokemon_last_state.items()
        if name not in fainted
    ]

    opponent_side = "p2" if side == "p1" else "p1"
    opponent_state = timeline[turn_index].get(f"{opponent_side}_pokemon_state", {}) or {}
    opponent_disabled = opponent_state.get("status", "nostatus") in {"slp", "frz"}

    roles = {
        role: sum(1 for name, _, _ in alive_pokemon if name in pokemon_set)
        for role, pokemon_set in ROLE_TAXONOMY.items()
    }

    return {
        "alive_pokemon": alive_pokemon,
        "opponent_disabled": opponent_disabled,
        "roles": roles,
    }


def _extract_damage_features(row):
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {
            "p1_total_damage_dealt": 0,
            "p2_total_damage_dealt": 0,
            "p1_avg_damage_per_turn": 0,
            "p2_avg_damage_per_turn": 0,
            "p1_damage_variance": 0,
            "p2_damage_variance": 0,
            "p1_ko_efficiency": 0,
            "p2_ko_efficiency": 0,
            "p1_early_damage_rate": 0,
            "p2_early_damage_rate": 0,
            "p1_late_damage_rate": 0,
            "p2_late_damage_rate": 0,
            "p1_max_single_hit": 0,
            "p2_max_single_hit": 0,
            "p1_ohko_count": 0,
            "p2_ohko_count": 0,
        }

    p1_damage_dealt = []
    p2_damage_dealt = []

    p1_early_damage = []
    p2_early_damage = []
    p1_late_damage = []
    p2_late_damage = []

    p1_ohko = 0
    p2_ohko = 0

    prev_p1_hp = None
    prev_p2_hp = None
    prev_p1_pokemon = None
    prev_p2_pokemon = None

    for t in timeline[:30]:
        turn = int(t.get("turn", 0))

        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}

        p1_hp = p1_state.get("hp_pct", 1.0)
        p2_hp = p2_state.get("hp_pct", 1.0)

        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()

        if prev_p1_hp is not None and prev_p1_pokemon == p1_name and p1_name:
            damage_to_p1 = max(0, prev_p1_hp - p1_hp)
            if damage_to_p1 > 0:
                p2_damage_dealt.append(damage_to_p1)

                if prev_p1_hp > 0.9 and p1_hp == 0:
                    p2_ohko += 1

                if turn <= 10:
                    p2_early_damage.append(damage_to_p1)
                elif turn >= 20:
                    p2_late_damage.append(damage_to_p1)

        if prev_p2_hp is not None and prev_p2_pokemon == p2_name and p2_name:
            damage_to_p2 = max(0, prev_p2_hp - p2_hp)
            if damage_to_p2 > 0:
                p1_damage_dealt.append(damage_to_p2)

                if prev_p2_hp > 0.9 and p2_hp == 0:
                    p1_ohko += 1

                if turn <= 10:
                    p1_early_damage.append(damage_to_p2)
                elif turn >= 20:
                    p1_late_damage.append(damage_to_p2)

        prev_p1_hp = p1_hp
        prev_p2_hp = p2_hp
        prev_p1_pokemon = p1_name
        prev_p2_pokemon = p2_name

    p1_total = sum(p1_damage_dealt)
    p2_total = sum(p2_damage_dealt)

    p1_avg = np.mean(p1_damage_dealt) if p1_damage_dealt else 0
    p2_avg = np.mean(p2_damage_dealt) if p2_damage_dealt else 0

    p1_var = np.var(p1_damage_dealt) if len(p1_damage_dealt) > 1 else 0
    p2_var = np.var(p2_damage_dealt) if len(p2_damage_dealt) > 1 else 0

    p1_kos = max(1, _count_kos_dealt(row, "p1"))
    p2_kos = max(1, _count_kos_dealt(row, "p2"))

    p1_efficiency = p1_total / p1_kos
    p2_efficiency = p2_total / p2_kos

    p1_early_rate = np.mean(p1_early_damage) if p1_early_damage else 0
    p2_early_rate = np.mean(p2_early_damage) if p2_early_damage else 0
    p1_late_rate = np.mean(p1_late_damage) if p1_late_damage else 0
    p2_late_rate = np.mean(p2_late_damage) if p2_late_damage else 0

    p1_max_hit = max(p1_damage_dealt) if p1_damage_dealt else 0
    p2_max_hit = max(p2_damage_dealt) if p2_damage_dealt else 0

    return {
        "p1_total_damage_dealt": p1_total,
        "p2_total_damage_dealt": p2_total,
        "p1_avg_damage_per_turn": p1_avg,
        "p2_avg_damage_per_turn": p2_avg,
        "p1_damage_variance": p1_var,
        "p2_damage_variance": p2_var,
        "p1_ko_efficiency": p1_efficiency,
        "p2_ko_efficiency": p2_efficiency,
        "p1_early_damage_rate": p1_early_rate,
        "p2_early_damage_rate": p2_early_rate,
        "p1_late_damage_rate": p1_late_rate,
        "p2_late_damage_rate": p2_late_rate,
        "p1_max_single_hit": p1_max_hit,
        "p2_max_single_hit": p2_max_hit,
        "p1_ohko_count": p1_ohko,
        "p2_ohko_count": p2_ohko,
    }


def _extract_move_quality_features(row):
    timeline = row.get("battle_timeline", []) or []
    if len(timeline) < 2:
        return {
            "p1_errors_count": 0,
            "p2_errors_count": 0,
            "p1_blunders_count": 0,
            "p2_blunders_count": 0,
            "p1_mean_negative_delta": 0,
            "p2_mean_negative_delta": 0,
            "p1_blunders_early": 0,
            "p2_blunders_early": 0,
            "p1_blunders_late": 0,
            "p2_blunders_late": 0,
        }

    counters = {
        "p1": {
            "deltas": [],
            "errors": 0,
            "blunders": 0,
            "blunders_early": 0,
            "blunders_late": 0,
        },
        "p2": {
            "deltas": [],
            "errors": 0,
            "blunders": 0,
            "blunders_early": 0,
            "blunders_late": 0,
        },
    }

    for i in range(len(timeline) - 1):
        turn = int(timeline[i].get("turn", 0))

        snapshots = {
            side: (
                _get_state_snapshot(timeline, i, side),
                _get_state_snapshot(timeline, i + 1, side),
            )
            for side in ["p1", "p2"]
        }

        if not all(pre and post for pre, post in snapshots.values()):
            continue

        for side in ["p1", "p2"]:
            pre, post = snapshots[side]
            delta = _calculate_position_score(post) - _calculate_position_score(pre)
            counters[side]["deltas"].append(delta)

            if delta <= ERROR_THRESHOLD:
                counters[side]["errors"] += 1
                if delta <= BLUNDER_THRESHOLD:
                    counters[side]["blunders"] += 1
                    if turn <= 10:
                        counters[side]["blunders_early"] += 1
                    elif turn >= 21:
                        counters[side]["blunders_late"] += 1

    return {
        "p1_errors_count": counters["p1"]["errors"],
        "p2_errors_count": counters["p2"]["errors"],
        "p1_blunders_count": counters["p1"]["blunders"],
        "p2_blunders_count": counters["p2"]["blunders"],
        "p1_mean_negative_delta": np.mean(
            [d for d in counters["p1"]["deltas"] if d < 0] or [0]
        ),
        "p2_mean_negative_delta": np.mean(
            [d for d in counters["p2"]["deltas"] if d < 0] or [0]
        ),
        "p1_blunders_early": counters["p1"]["blunders_early"],
        "p2_blunders_early": counters["p2"]["blunders_early"],
        "p1_blunders_late": counters["p1"]["blunders_late"],
        "p2_blunders_late": counters["p2"]["blunders_late"],
    }


def _extract_momentum_features(row):
    timeline = row.get("battle_timeline", []) or []
    if len(timeline) < 3:
        return {
            "p1_hp_momentum": 0,
            "p2_hp_momentum": 0,
            "hp_momentum_diff": 0,
            "largest_hp_swing": 0,
        }

    p1_hp_changes = []
    p2_hp_changes = []

    prev_p1_hp = prev_p2_hp = 1.0
    for t in timeline[:30]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}

        p1_hp = p1_state.get("hp_pct", 1.0)
        p2_hp = p2_state.get("hp_pct", 1.0)

        p1_hp_changes.append(p1_hp - prev_p1_hp)
        p2_hp_changes.append(p2_hp - prev_p2_hp)

        prev_p1_hp = p1_hp
        prev_p2_hp = p2_hp

    weights = np.exp(np.linspace(-2, 0, len(p1_hp_changes)))
    p1_momentum = (
        np.average(p1_hp_changes, weights=weights) if p1_hp_changes else 0
    )
    p2_momentum = (
        np.average(p2_hp_changes, weights=weights) if p2_hp_changes else 0
    )

    window_size = 5
    if len(p1_hp_changes) >= window_size * 2:
        p1_windows = [
            sum(p1_hp_changes[i : i + window_size])
            for i in range(0, len(p1_hp_changes) - window_size, window_size)
        ]
        p2_windows = [
            sum(p2_hp_changes[i : i + window_size])
            for i in range(0, len(p2_hp_changes) - window_size, window_size)
        ]

        largest_swing = (
            max(abs(p1_windows[i] - p1_windows[i - 1]) for i in range(1, len(p1_windows)))
            if len(p1_windows) > 1
            else 0
        )
    else:
        largest_swing = 0

    return {
        "p1_hp_momentum": p1_momentum,
        "p2_hp_momentum": p2_momentum,
        "hp_momentum_diff": p1_momentum - p2_momentum,
        "largest_hp_swing": largest_swing,
    }


def _get_pokemon_hp_at_turn(timeline, target_turn, side, pokemon_name):
    for t in reversed(timeline):
        if int(t.get("turn", 0)) > target_turn:
            continue
        state = t.get(f"{side}_pokemon_state", {}) or {}
        name = (state.get("name") or "").lower()
        if name == pokemon_name:
            return state.get("hp_pct", 0)
    return 0


def _extract_critical_pokemon_features(row):
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {
            "p1_critical_alive": 0,
            "p2_critical_alive": 0,
            "critical_alive_diff": 0,
            "p1_critical_healthy": 0,
            "p2_critical_healthy": 0,
        }

    CRITICAL_MONS = {"chansey", "tauros", "alakazam", "starmie", "zapdos"}

    turns = [int(t.get("turn", 0)) for t in timeline]
    target_turn = min(30, max(turns)) if turns else 30

    p1_team = set()
    p2_team = set()
    p1_fainted = set()
    p2_fainted = set()

    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break

        for side, team_set, fainted_set in [
            ("p1", p1_team, p1_fainted),
            ("p2", p2_team, p2_fainted),
        ]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            name = (state.get("name") or "").lower()
            if name:
                team_set.add(name)
                if state.get("hp_pct", 1.0) == 0:
                    fainted_set.add(name)

    p1_alive_set = (p1_team & CRITICAL_MONS) - p1_fainted
    p2_alive_set = (p2_team & CRITICAL_MONS) - p2_fainted

    p1_critical_alive = len(p1_alive_set)
    p2_critical_alive = len(p2_alive_set)

    p1_critical_healthy = sum(
        1
        for p in p1_alive_set
        if _get_pokemon_hp_at_turn(timeline, target_turn, "p1", p) > 0.5
    )
    p2_critical_healthy = sum(
        1
        for p in p2_alive_set
        if _get_pokemon_hp_at_turn(timeline, target_turn, "p2", p) > 0.5
    )

    return {
        "p1_critical_alive": p1_critical_alive,
        "p2_critical_alive": p2_critical_alive,
        "critical_alive_diff": p1_critical_alive - p2_critical_alive,
        "p1_critical_healthy": p1_critical_healthy,
        "p2_critical_healthy": p2_critical_healthy,
    }


def _extract_early_game_features(row):
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {
            "p1_early_damage": 0,
            "p2_early_damage": 0,
            "p1_early_switches": 0,
            "p2_early_switches": 0,
            "early_switch_diff": 0,
        }

    EARLY_TURNS = 10

    p1_damage = p2_damage = 0
    p1_switches = p2_switches = 0
    p1_prev_pokemon = p2_prev_pokemon = None

    for t in timeline[:EARLY_TURNS]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}

        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()

        if p1_prev_pokemon and p1_name != p1_prev_pokemon:
            p1_switches += 1
        if p2_prev_pokemon and p2_name != p2_prev_pokemon:
            p2_switches += 1

        p1_prev_pokemon = p1_name
        p2_prev_pokemon = p2_name

        p1_damage += 1.0 - p2_state.get("hp_pct", 1.0)
        p2_damage += 1.0 - p1_state.get("hp_pct", 1.0)

    return {
        "p1_early_damage": p1_damage / EARLY_TURNS,
        "p2_early_damage": p2_damage / EARLY_TURNS,
        "p1_early_switches": p1_switches,
        "p2_early_switches": p2_switches,
        "early_switch_diff": p1_switches - p2_switches,
    }


def _extract_boost_features(row):
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {
            "p1_max_boosts": 0,
            "p2_max_boosts": 0,
            "max_boosts_diff": 0,
            "p1_boost_turns": 0,
            "p2_boost_turns": 0,
            "boost_turns_diff": 0,
        }

    p1_max_boosts = p2_max_boosts = 0
    p1_boost_turns = p2_boost_turns = 0

    for t in timeline[:30]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}

        p1_boosts = p1_state.get("boosts", {})
        p2_boosts = p2_state.get("boosts", {})

        if isinstance(p1_boosts, dict):
            p1_total = sum(abs(v) for v in p1_boosts.values())
            p1_max_boosts = max(p1_max_boosts, p1_total)
            if p1_total > 0:
                p1_boost_turns += 1

        if isinstance(p2_boosts, dict):
            p2_total = sum(abs(v) for v in p2_boosts.values())
            p2_max_boosts = max(p2_max_boosts, p2_total)
            if p2_total > 0:
                p2_boost_turns += 1

    return {
        "p1_max_boosts": p1_max_boosts,
        "p2_max_boosts": p2_max_boosts,
        "max_boosts_diff": p1_max_boosts - p2_max_boosts,
        "p1_boost_turns": p1_boost_turns,
        "p2_boost_turns": p2_boost_turns,
        "boost_turns_diff": p1_boost_turns - p2_boost_turns,
    }


def _extract_static_features(row):
    p1_team = [p.lower() for p in row.get("p1_team", []) if p]
    p2_team = [p.lower() for p in row.get("p2_team", []) if p]

    if not p1_team or not p2_team:
        return {
            "p1_team_value": 0,
            "p2_team_value": 0,
            "p1_type_coverage": 0,
            "p2_type_coverage": 0,
            "p1_speed_control": 0,
            "p2_speed_control": 0,
            "p1_physical_threats": 0,
            "p2_physical_threats": 0,
            "p1_special_threats": 0,
            "p2_special_threats": 0,
            "p1_walls": 0,
            "p2_walls": 0,
        }

    p1_value = sum(POKEMON_VALUES.get(p, 0.6) for p in p1_team) / len(p1_team)
    p2_value = sum(POKEMON_VALUES.get(p, 0.6) for p in p2_team) / len(p2_team)

    p1_coverage = len(
        {
            adv
            for p in p1_team
            for t in POKEMON_TYPES.get(p, [])
            for adv in TYPE_ADVANTAGES.get(t, set())
        }
    )
    p2_coverage = len(
        {
            adv
            for p in p2_team
            for t in POKEMON_TYPES.get(p, [])
            for adv in TYPE_ADVANTAGES.get(t, set())
        }
    )

    p1_fast = sum(1 for p in p1_team if SPEED_TIERS.get(p, 0) >= 100)
    p2_fast = sum(1 for p in p2_team if SPEED_TIERS.get(p, 0) >= 100)

    p1_roles = {
        role: sum(1 for p in p1_team if p in poke_set)
        for role, poke_set in ROLE_TAXONOMY.items()
    }
    p2_roles = {
        role: sum(1 for p in p2_team if p in poke_set)
        for role, poke_set in ROLE_TAXONOMY.items()
    }

    return {
        "p1_team_value": p1_value,
        "p2_team_value": p2_value,
        "p1_type_coverage": p1_coverage,
        "p2_type_coverage": p2_coverage,
        "p1_speed_control": p1_fast,
        "p2_speed_control": p2_fast,
        "p1_physical_threats": p1_roles.get("breaker_phys", 0),
        "p2_physical_threats": p2_roles.get("breaker_phys", 0),
        "p1_special_threats": p1_roles.get("breaker_spec", 0),
        "p2_special_threats": p2_roles.get("breaker_spec", 0),
        "p1_walls": p1_roles.get("wall_phys", 0) + p1_roles.get("wall_spec", 0),
        "p2_walls": p2_roles.get("wall_phys", 0) + p2_roles.get("wall_spec", 0),
    }


# =============================================================================
# _make_... FEATURE FONKSİYONLARI
# =============================================================================

def _make_status_features(row):
    events = _collect_status_events(row)

    early_slp_highvalue_opponent = any(
        e["turn"] <= EARLY_TURN_THRESHOLD
        and e["status"] == "slp"
        and e["target"] in HIGHVALUE_SLEEP_TARGETS
        for e in events
    )
    early_par_on_tauros_or_psychic = any(
        e["turn"] <= EARLY_TURN_THRESHOLD
        and e["status"] == "par"
        and e["target"] in HIGHVALUE_PARALYZE_TARGETS
        for e in events
    )
    freeze_on_special_wall = any(
        e["status"] == "frz" and e["target"] in SPECIAL_WALLS_FOR_FREEZE for e in events
    )
    burn_on_physical_titan = any(
        e["status"] == "brn" and e["target"] in PHYSICAL_TITANS_FOR_BURN for e in events
    )

    first_p1 = _first_status(events, "p1")
    first_status_turn_p1 = first_p1["turn"] if first_p1 else -1

    hv_pool_all = (
        HIGHVALUE_SLEEP_TARGETS
        | HIGHVALUE_PARALYZE_TARGETS
        | SPECIAL_WALLS_FOR_FREEZE
        | PHYSICAL_TITANS_FOR_BURN
    )
    first_status_is_highvalue_p1 = int(first_p1["target"] in hv_pool_all) if first_p1 else 0

    p1_highvalue_status_count = _count_highvalue(events, "p1", hv_pool_all)
    p2_highvalue_status_count = _count_highvalue(events, "p2", hv_pool_all)

    early_diff, late_diff = _turns_disabled_diff_temporal(row)

    return pd.Series(
        {
            "early_slp_highvalue_opponent": int(early_slp_highvalue_opponent),
            "early_par_on_tauros_or_psychic": int(early_par_on_tauros_or_psychic),
            "freeze_on_special_wall": int(freeze_on_special_wall),
            "burn_on_physical_titan": int(burn_on_physical_titan),
            "first_status_turn_p1": first_status_turn_p1,
            "first_status_is_highvalue_p1": first_status_is_highvalue_p1,
            "status_diff_highvalue": p1_highvalue_status_count - p2_highvalue_status_count,
            "turns_disabled_diff": _turns_disabled_diff(row),
            "late_game_status_swing": _late_game_flag(events),
            "p1_highvalue_status_count": p1_highvalue_status_count,
            "p2_highvalue_status_count": p2_highvalue_status_count,
            "turns_disabled_diff_early": early_diff,
            "turns_disabled_diff_late": late_diff,
            "status_diff_highvalue_w": _status_diff_highvalue_w(row),
            "late_game_status_swing_v2": _late_game_status_swing_v2(row),
            "turns_disabled_diff_w_decay": _turns_disabled_diff_w_decay(row),
        },
        dtype="float64",
    )


def _make_endgame_features(row):
    p1_kos, p2_kos, p1_surv, p2_surv, p1_eff, p2_eff = _extract_endgame_state(row)

    return pd.Series(
        {
            "p1_kos_30": min(6, max(0, p1_kos)),
            "p2_kos_30": min(6, max(0, p2_kos)),
            "ko_diff_30": p2_kos - p1_kos,
            "p1_survivors_30": p1_surv,
            "p2_survivors_30": p2_surv,
            "survivor_diff_30": p1_surv - p2_surv,
            "active_effects_count_p1_end": len(p1_eff),
            "active_effects_count_p2_end": len(p2_eff),
            "active_effects_diff_end": len(p1_eff) - len(p2_eff),
            "active_effects_weighted_p1_end": sum(
                EFFECT_WEIGHTS.get(eff, 1.0) for eff in p1_eff
            ),
            "active_effects_weighted_p2_end": sum(
                EFFECT_WEIGHTS.get(eff, 1.0) for eff in p2_eff
            ),
            "active_effects_weighted_diff_end": sum(
                EFFECT_WEIGHTS.get(eff, 1.0) for eff in p1_eff
            )
            - sum(EFFECT_WEIGHTS.get(eff, 1.0) for eff in p2_eff),
        },
        dtype="float64",
    )


def _make_role_features(row):
    role_counts = _extract_survivor_roles(row)
    features = {}

    for role, (p1_count, p2_count) in role_counts.items():
        features[f"p1_rolecount_{role}_end"] = p1_count
        features[f"p2_rolecount_{role}_end"] = p2_count
        features[f"rolecount_{role}_diff_end"] = p1_count - p2_count

    return pd.Series(features, dtype="float64")


def _make_damage_features(row):
    stats = _extract_damage_features(row)

    return pd.Series(
        {
            "p1_total_damage_dealt": stats["p1_total_damage_dealt"],
            "p2_total_damage_dealt": stats["p2_total_damage_dealt"],
            "total_damage_diff": stats["p1_total_damage_dealt"]
            - stats["p2_total_damage_dealt"],
            "p1_avg_damage_per_turn": stats["p1_avg_damage_per_turn"],
            "p2_avg_damage_per_turn": stats["p2_avg_damage_per_turn"],
            "avg_damage_diff": stats["p1_avg_damage_per_turn"]
            - stats["p2_avg_damage_per_turn"],
            "p1_damage_variance": stats["p1_damage_variance"],
            "p2_damage_variance": stats["p2_damage_variance"],
            "damage_variance_diff": stats["p1_damage_variance"]
            - stats["p2_damage_variance"],
            "p1_ko_efficiency": stats["p1_ko_efficiency"],
            "p2_ko_efficiency": stats["p2_ko_efficiency"],
            "ko_efficiency_diff": stats["p1_ko_efficiency"]
            - stats["p2_ko_efficiency"],
            "p1_early_damage_rate": stats["p1_early_damage_rate"],
            "p2_early_damage_rate": stats["p2_early_damage_rate"],
            "early_damage_diff": stats["p1_early_damage_rate"]
            - stats["p2_early_damage_rate"],
            "p1_late_damage_rate": stats["p1_late_damage_rate"],
            "p2_late_damage_rate": stats["p2_late_damage_rate"],
            "late_damage_diff": stats["p1_late_damage_rate"]
            - stats["p2_late_damage_rate"],
            "p1_max_single_hit": stats["p1_max_single_hit"],
            "p2_max_single_hit": stats["p2_max_single_hit"],
            "max_hit_diff": stats["p1_max_single_hit"] - stats["p2_max_single_hit"],
            "p1_ohko_count": stats["p1_ohko_count"],
            "p2_ohko_count": stats["p2_ohko_count"],
            "ohko_diff": stats["p1_ohko_count"] - stats["p2_ohko_count"],
            "p1_damage_consistency": stats["p1_avg_damage_per_turn"]
            / (stats["p1_damage_variance"] + 0.01),
            "p2_damage_consistency": stats["p2_avg_damage_per_turn"]
            / (stats["p2_damage_variance"] + 0.01),
            "damage_consistency_diff": stats["p1_avg_damage_per_turn"]
            / (stats["p1_damage_variance"] + 0.01)
            - stats["p2_avg_damage_per_turn"]
            / (stats["p2_damage_variance"] + 0.01),
        },
        dtype="float64",
    )


def _make_hp_distribution_features(row):
    stats = _calculate_hp_distribution_features(row)

    timeline = row.get("battle_timeline", []) or []
    p1_alive = p2_alive = 0

    if timeline:
        turns = [int(t.get("turn", 0)) for t in timeline]
        if turns:
            target_turn = min(30, max(turns))
            p1_fainted, p2_fainted = set(), set()
            p1_team, p2_team = set(), set()

            for t in timeline:
                turn = int(t.get("turn", 0))
                if turn > target_turn:
                    break

                for side, fainted_set, team_set in [
                    ("p1", p1_fainted, p1_team),
                    ("p2", p2_fainted, p2_team),
                ]:
                    state = t.get(f"{side}_pokemon_state", {}) or {}
                    name = (state.get("name") or "").lower()
                    if name:
                        team_set.add(name)
                        if state.get("hp_pct", 0) == 0:
                            fainted_set.add(name)

            p1_alive = len(p1_team - p1_fainted)
            p2_alive = len(p2_team - p2_fainted)

    alive_count_diff = p1_alive - p2_alive

    diffs = {
        "avg_hp_alive_diff": np.clip(
            stats["p1_avg_hp"] - stats["p2_avg_hp"], -1.0, 1.0
        ),
        "std_hp_alive_diff": np.clip(
            stats["p1_std_hp"] - stats["p2_std_hp"], -0.5, 0.5
        ),
        "cv_hp_alive_diff": stats["p1_cv_hp"] - stats["p2_cv_hp"],
        "median_hp_alive_diff": stats["p1_median_hp"] - stats["p2_median_hp"],
        "dispersion_flag_diff": stats["p1_high_dispersion"]
        - stats["p2_high_dispersion"],
        "weighted_avg_hp_diff": np.clip(
            stats["p1_weighted_avg_hp"] - stats["p2_weighted_avg_hp"], -1.0, 1.0
        ),
        "effective_avg_hp_diff": np.clip(
            stats["p1_effective_avg_hp"] - stats["p2_effective_avg_hp"], -1.0, 1.0
        ),
    }

    return pd.Series(
        {
            "p1_avg_hp_alive": stats["p1_avg_hp"],
            "p2_avg_hp_alive": stats["p2_avg_hp"],
            "p1_std_hp_alive": stats["p1_std_hp"],
            "p2_std_hp_alive": stats["p2_std_hp"],
            **diffs,
            "std_hp_x_alive_diff": diffs["std_hp_alive_diff"] * alive_count_diff,
        },
        dtype="float64",
    )


def _make_status_t30_features(row):
    sc = _extract_status_at_t30(row)

    return pd.Series(
        {
            "p1_statused_alive_end": sc["p1_statused_alive"],
            "p2_statused_alive_end": sc["p2_statused_alive"],
            "statused_alive_end_diff": sc["p1_statused_alive"]
            - sc["p2_statused_alive"],
            "p1_sleepers_end": sc["p1_sleepers"],
            "p2_sleepers_end": sc["p2_sleepers"],
            "sleepers_end_diff": sc["p1_sleepers"] - sc["p2_sleepers"],
            "p1_freezes_end": sc["p1_freezes"],
            "p2_freezes_end": sc["p2_freezes"],
            "freezes_end_diff": sc["p1_freezes"] - sc["p2_freezes"],
            "p1_paras_end": sc["p1_paras"],
            "p2_paras_end": sc["p2_paras"],
            "paras_end_diff": sc["p1_paras"] - sc["p2_paras"],
        },
        dtype="float64",
    )


def _make_move_quality_features(row):
    q = _extract_move_quality_features(row)

    return pd.Series(
        {
            "p1_errors_count": q["p1_errors_count"],
            "p2_errors_count": q["p2_errors_count"],
            "errors_diff": q["p1_errors_count"] - q["p2_errors_count"],
            "p1_blunders_count": q["p1_blunders_count"],
            "p2_blunders_count": q["p2_blunders_count"],
            "blunders_diff": q["p1_blunders_count"] - q["p2_blunders_count"],
            "p1_mean_negative_delta": q["p1_mean_negative_delta"],
            "p2_mean_negative_delta": q["p2_mean_negative_delta"],
            "negative_delta_diff": q["p1_mean_negative_delta"]
            - q["p2_mean_negative_delta"],
            "p1_blunders_early": q["p1_blunders_early"],
            "p2_blunders_early": q["p2_blunders_early"],
            "blunders_early_diff": q["p1_blunders_early"]
            - q["p2_blunders_early"],
            "p1_blunders_late": q["p1_blunders_late"],
            "p2_blunders_late": q["p2_blunders_late"],
            "blunders_late_diff": q["p1_blunders_late"]
            - q["p2_blunders_late"],
        },
        dtype="float64",
    )


def _make_static_features(row):
    stats = _extract_static_features(row)

    return pd.Series(
        {
            "p1_team_value": stats["p1_team_value"],
            "p2_team_value": stats["p2_team_value"],
            "team_value_diff": stats["p1_team_value"] - stats["p2_team_value"],
            "p1_type_coverage": stats["p1_type_coverage"],
            "p2_type_coverage": stats["p2_type_coverage"],
            "type_coverage_diff": stats["p1_type_coverage"]
            - stats["p2_type_coverage"],
            "p1_speed_control": stats["p1_speed_control"],
            "p2_speed_control": stats["p2_speed_control"],
            "speed_control_diff": stats["p1_speed_control"]
            - stats["p2_speed_control"],
            "p1_physical_threats": stats["p1_physical_threats"],
            "p2_physical_threats": stats["p2_physical_threats"],
            "physical_threats_diff": stats["p1_physical_threats"]
            - stats["p2_physical_threats"],
            "p1_special_threats": stats["p1_special_threats"],
            "p2_special_threats": stats["p2_special_threats"],
            "special_threats_diff": stats["p1_special_threats"]
            - stats["p2_special_threats"],
            "p1_walls": stats["p1_walls"],
            "p2_walls": stats["p2_walls"],
            "walls_diff": stats["p1_walls"] - stats["p2_walls"],
            "offensive_pressure_p1": stats["p1_physical_threats"]
            + stats["p1_special_threats"],
            "offensive_pressure_p2": stats["p2_physical_threats"]
            + stats["p2_special_threats"],
            "offensive_pressure_diff": (stats["p1_physical_threats"]
            + stats["p1_special_threats"])
            - (stats["p2_physical_threats"] + stats["p2_special_threats"]),
            "p1_balance": abs(
                stats["p1_physical_threats"] - stats["p1_special_threats"]
            ),
            "p2_balance": abs(
                stats["p2_physical_threats"] - stats["p2_special_threats"]
            ),
            "balance_diff": abs(
                stats["p1_physical_threats"] - stats["p1_special_threats"]
            )
            - abs(stats["p2_physical_threats"] - stats["p2_special_threats"]),
        },
        dtype="float64",
    )


def _make_interaction_features(row):
    g = row.get
    out = {}

    # MOMENTUM × MATERIAL
    out["momentum_x_survivors"] = g("hp_momentum_diff", 0) * g("survivor_diff_30", 0)
    out["momentum_x_ko"] = g("hp_momentum_diff", 0) * g("ko_diff_30", 0)

    # CRITICAL POKÉMON
    out["critical_x_hp_quality"] = g("critical_alive_diff", 0) * g(
        "avg_hp_alive_diff", 0
    )
    out["critical_x_status_free"] = g("critical_alive_diff", 0) * (
        -g("statused_alive_end_diff", 0)
    )

    # EARLY → LATE
    out["early_damage_to_ko"] = g("early_damage_diff", 0) * g("ko_diff_30", 0)
    out["early_switches_x_survivors"] = g("early_switch_diff", 0) * g(
        "survivor_diff_30", 0
    )

    # BOOSTS × SURVIVORS
    out["boosts_x_alive"] = g("max_boosts_diff", 0) * g("survivor_diff_30", 0)
    out["boosts_x_hp"] = g("max_boosts_diff", 0) * g("avg_hp_alive_diff", 0)

    # HP DISPERSION × ROLE
    out["hp_dispersion_x_breakers"] = g("std_hp_alive_diff", 0) * g(
        "rolecount_breaker_spec_diff_end", 0
    )

    # MOVE QUALITY × MATERIAL
    out["blunders_x_ko_loss"] = g("blunders_diff", 0) * (-g("ko_diff_30", 0))
    out["errors_x_hp_loss"] = g("errors_diff", 0) * (-g("avg_hp_alive_diff", 0))

    # TEAM COMP × EXECUTION
    out["team_value_x_errors"] = g("team_value_diff", 0) * (-g("errors_diff", 0))
    out["speed_control_x_momentum"] = g("speed_control_diff", 0) * g(
        "hp_momentum_diff", 0
    )

    # WALLS UNDER PRESSURE
    out["walls_x_status"] = g("walls_diff", 0) * g("statused_alive_end_diff", 0)
    out["walls_x_hp"] = g("walls_diff", 0) * g("avg_hp_alive_diff", 0)

    # STATUS AND EFFECTS
    out["ko_status_interaction"] = g("ko_diff_30", 0) * g("status_diff_highvalue", 0)
    out["survivor_disabled_interaction"] = g("survivor_diff_30", 0) * g(
        "turns_disabled_diff", 0
    )
    out["effects_momentum_interaction"] = g(
        "active_effects_weighted_diff_end", 0
    ) * g("late_game_status_swing", 0)

    # ENDGAME
    out["hp_p1_x_survivors"] = g("p1_avg_hp_alive", 0) * g("p1_survivors_30", 0)
    out["hp_p2_x_survivors"] = g("p2_avg_hp_alive", 0) * g("p2_survivors_30", 0)
    out["hp_p1_x_kos"] = g("p1_avg_hp_alive", 0) * g("p1_kos_30", 0)
    out["hp_p2_x_kos"] = g("p2_avg_hp_alive", 0) * g("p2_kos_30", 0)
    out["endgame_result_consistency"] = g("ko_diff_30", 0) * g("survivor_diff_30", 0)
    out["disabled_to_conversion"] = g("ko_diff_30", 0) * g(
        "turns_disabled_diff_w_decay", 0
    )
    out["control_x_status_pressure"] = g(
        "turns_disabled_diff_w_decay", 0
    ) * g("statused_alive_end_diff", 0)
    out["durable_advantage_compound"] = g("effective_avg_hp_diff", 0) * g(
        "survivor_diff_30", 0
    )
    out["sleep_power_factor"] = g("sleepers_end_diff", 0) * g(
        "turns_disabled_diff_w_decay", 0
    )
    out["freeze_breakers_lock"] = g("freezes_end_diff", 0) * g(
        "p2_rolecount_breaker_phys_end", 0
    )
    out["neuter_breakers_with_status"] = g("rolecount_breaker_phys_diff_end", 0) * g(
        "statused_alive_end_diff", 0
    )
    out["control_closure_link"] = (
        g("survivor_diff_30", 0) * g("turns_disabled_diff", 0) * g("ko_diff_30", 0)
    )
    out["opp_hp_under_status"] = g("p2_avg_hp_alive", 0) * g(
        "p2_statused_alive_end", 0
    )

    # DAMAGE INTERACTIONS (NEW)
    out["damage_x_ko"] = g("total_damage_diff", 0) * g("ko_diff_30", 0)
    out["damage_efficiency_x_survivors"] = g("ko_efficiency_diff", 0) * g(
        "survivor_diff_30", 0
    )
    out["burst_damage_x_critical"] = g("max_hit_diff", 0) * g(
        "critical_alive_diff", 0
    )
    out["early_damage_x_status"] = g("early_damage_diff", 0) * g(
        "status_diff_highvalue", 0
    )
    out["late_damage_x_control"] = g("late_damage_diff", 0) * g(
        "turns_disabled_diff", 0
    )
    out["ohko_x_momentum"] = g("ohko_diff", 0) * g("hp_momentum_diff", 0)
    out["damage_consistency_x_walls"] = g("damage_consistency_diff", 0) * g(
        "walls_diff", 0
    )

    return pd.Series(out, dtype="float64")


# =============================================================================
# ANA FUNKSİYON: build_feature_frames
# =============================================================================

def build_feature_frames(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Orijinal notebook'taki tüm feature extraction kısmını buraya topluyoruz.
    Dönüş:
      - train_df: [battle_id, player_won, features...]
      - test_df:  [battle_id, features...]
    """
    print("\n Extracting features...")

    # Status features
    train_status = train_df.apply(_make_status_features, axis=1)
    test_status = test_df.apply(_make_status_features, axis=1)

    # Endgame features
    train_endgame = train_df.apply(_make_endgame_features, axis=1)
    test_endgame = test_df.apply(_make_endgame_features, axis=1)

    # Role features
    train_roles = train_df.apply(_make_role_features, axis=1)
    test_roles = test_df.apply(_make_role_features, axis=1)

    # HP distribution
    train_hp = train_df.apply(_make_hp_distribution_features, axis=1)
    test_hp = test_df.apply(_make_hp_distribution_features, axis=1)

    # Status at T30
    train_status_t30 = train_df.apply(_make_status_t30_features, axis=1)
    test_status_t30 = test_df.apply(_make_status_t30_features, axis=1)

    # Move quality
    train_moves = train_df.apply(_make_move_quality_features, axis=1)
    test_moves = test_df.apply(_make_move_quality_features, axis=1)

    # Static team
    train_static = train_df.apply(_make_static_features, axis=1)
    test_static = test_df.apply(_make_static_features, axis=1)

    # Momentum
    train_momentum = train_df.apply(
        lambda r: pd.Series(_extract_momentum_features(r)), axis=1
    )
    test_momentum = test_df.apply(
        lambda r: pd.Series(_extract_momentum_features(r)), axis=1
    )

    # Critical pokemon
    train_critical = train_df.apply(
        lambda r: pd.Series(_extract_critical_pokemon_features(r)), axis=1
    )
    test_critical = test_df.apply(
        lambda r: pd.Series(_extract_critical_pokemon_features(r)), axis=1
    )

    # Early game
    train_early = train_df.apply(
        lambda r: pd.Series(_extract_early_game_features(r)), axis=1
    )
    test_early = test_df.apply(
        lambda r: pd.Series(_extract_early_game_features(r)), axis=1
    )

    # Boosts
    train_boosts = train_df.apply(
        lambda r: pd.Series(_extract_boost_features(r)), axis=1
    )
    test_boosts = test_df.apply(
        lambda r: pd.Series(_extract_boost_features(r)), axis=1
    )

    # Damage
    train_damage = train_df.apply(_make_damage_features, axis=1)
    test_damage = test_df.apply(_make_damage_features, axis=1)

    # Interaction features
    train_interactions = (
        train_df.apply(_make_interaction_features, axis=1).astype("float64")
    )
    test_interactions = (
        test_df.apply(_make_interaction_features, axis=1).astype("float64")
    )

    # Hepsini birleştir
    train_features = pd.concat(
        [
            train_status,
            train_endgame,
            train_roles,
            train_hp,
            train_status_t30,
            train_moves,
            train_static,
            train_momentum,
            train_critical,
            train_early,
            train_boosts,
            train_damage,
            train_interactions,
        ],
        axis=1,
    )

    test_features = pd.concat(
        [
            test_status,
            test_endgame,
            test_roles,
            test_hp,
            test_status_t30,
            test_moves,
            test_static,
            test_momentum,
            test_critical,
            test_early,
            test_boosts,
            test_damage,
            test_interactions,
        ],
        axis=1,
    )

    # Final dataframes
    train_df_out = pd.concat(
        [train_df[[COL_ID, COL_TARGET]].reset_index(drop=True),
         train_features.reset_index(drop=True)],
        axis=1,
    )
    test_df_out = pd.concat(
        [test_df[[COL_ID]].reset_index(drop=True),
         test_features.reset_index(drop=True)],
        axis=1,
    )

    print(f" Final feature count: {train_features.shape[1]}")
    return train_df_out, test_df_out
