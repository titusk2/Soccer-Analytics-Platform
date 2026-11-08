import numpy as np
from .ai_extraction import extract_factors_with_llm
from .scoring import get_player_score
from .ai_extraction import extract_factors_with_llm


allowed_factors = {
    "leadership",
    "decision_making",
    "work_rate",
    "tactical_fit",
    "press_resistance",
    "adaptability",
    "consistency",
    "injury_concern",
    "communication",
    "off_ball_movement"
}

factor_weights = {
    "leadership": 2.0,
    "decision_making": 2.0,
    "work_rate": 2.0,
    "tactical_fit": 3.0,
    "press_resistance": 2.5,
    "adaptability": 1.5,
    "consistency": 2.5,
    "injury_concern": 3.0,
    "communication": 1.5,
    "off_ball_movement": 2.0
}

def validate_factors(parsed_output):

    valid_factors = []

    for item in parsed_output.get(
        "factors",
        []
    ):

        factor = item.get("factor")

        if factor not in allowed_factors:
            continue

        direction = item.get(
            "direction",
            0
        )

        strength = item.get(
            "strength",
            0
        )

        confidence = item.get(
            "confidence",
            0
        )

        # Validate direction
        if direction not in [-1, 1]:
            continue

        # Keep strength/confidence between 0 and 1
        strength = min(
            max(float(strength), 0),
            1
        )

        confidence = min(
            max(float(confidence), 0),
            1
        )

        valid_factors.append({
            "factor": factor,
            "direction": direction,
            "strength": strength,
            "confidence": confidence
        })

    return valid_factors

def calculate_context_adjustments(
    factors
):

    adjustments = []

    for item in factors:

        factor = item["factor"]

        weight = factor_weights[
            factor
        ]

        adjustment = (
            weight
            * item["direction"]
            * item["strength"]
            * item["confidence"]
        )

        adjustments.append({
            "factor": factor,
            "direction": item["direction"],
            "strength": item["strength"],
            "confidence": item["confidence"],
            "weight": weight,
            "adjustment": adjustment
        })

    return adjustments

MAX_TOTAL_ADJUSTMENT = 10

def calculate_personalized_score(
    base_score,
    adjustments
):

    total_adjustment = sum(
        item["adjustment"]
        for item in adjustments
    )

    total_adjustment = np.clip(
        total_adjustment,
        -MAX_TOTAL_ADJUSTMENT,
        MAX_TOTAL_ADJUSTMENT
    )

    personalized_score = np.clip(
        base_score + total_adjustment,
        0,
        100
    )

    return (
        round(personalized_score, 1),
        round(total_adjustment, 2)
    )

def personalize_player_score(
    player_name,
    position,
    position_data,
    parsed_chatbot_output,
    season=None
):

    player = get_player_score(
        player_name,
        position,
        position_data,
        season=season
    )

    base_score = player[
        "performance_score"
    ]

    factors = validate_factors(
        parsed_chatbot_output
    )

    adjustments = (
        calculate_context_adjustments(
            factors
        )
    )

    personalized_score, total_adjustment = (
        calculate_personalized_score(
            base_score,
            adjustments
        )
    )

    return {
        "player": player_name,
        "position": position,
        "team": player["team"],
        "season": player["season"],
        "archetype": player["archetype"],
        "base_score": base_score,
        "personalized_score": personalized_score,
        "total_adjustment": total_adjustment,
        "factors": adjustments
    }

def evaluate_with_scouting_note(
    player_name,
    position,
    scouting_note,
    position_data,
    season=None,
    model="gpt-6-luna"
):

    parsed_output = extract_factors_with_llm(
        scouting_note,
        model=model
    )

    result = personalize_player_score(
        player_name,
        position,
        position_data,
        parsed_output,
        season=season
    )

    result["scouting_note"] = scouting_note
    result["llm_summary"] = parsed_output["summary"]

    return result
