import numpy as np

def score_confidence(completeness):

    if completeness >= 95:
        return "High"
    elif completeness >= 75:
        return "Moderate"
    elif completeness >= 50:
        return "Low"
    else:
        return "Insufficient"


#Player lookup function
def get_player_score(
    player_name,
    position,
    position_data,
    season=None
):

    dataframe = position_data[position]

    player_rows = dataframe[
        dataframe["player"].str.lower()
        == player_name.lower()
    ]

    if player_rows.empty:
        raise ValueError(
            f"{player_name} not found in {position} data."
        )

    if season is not None:

        season_rows = player_rows[
            player_rows["season"] == season
        ]

        if season_rows.empty:

            available_seasons = (
                player_rows["season"]
                .dropna()
                .sort_values()
                .unique()
            )

            raise ValueError(
                f"{player_name} does not have data "
                f"for season {season}. "
                f"Available seasons: {available_seasons}"
            )

        player_rows = season_rows

    player_row = (
        player_rows
        .sort_values(
            "season",
            ascending=False
        )
        .iloc[0]
    )

    return player_row


archetype_weights = {

    "Defensive / Holding Midfielder": {
        "attacking_score": 0.10,
        "creation_score": 0.10,
        "progression_score": 0.20,
        "possession_score": 0.25,
        "defending_score": 0.35
    },

    "Creative / Progressive Midfielder": {
        "attacking_score": 0.20,
        "creation_score": 0.30,
        "progression_score": 0.25,
        "possession_score": 0.15,
        "defending_score": 0.10
    },

    "Direct / Finishing Forward": {
        "attacking_score": 0.45,
        "creation_score": 0.20,
        "progression_score": 0.15,
        "possession_score": 0.10,
        "defending_score": 0.10
    },

    "Creative / Progressive Forward": {
        "attacking_score": 0.25,
        "creation_score": 0.35,
        "progression_score": 0.25,
        "possession_score": 0.10,
        "defending_score": 0.05
    },

    "Attacking / Progressive Defender": {
        "attacking_score": 0.10,
        "creation_score": 0.25,
        "progression_score": 0.30,
        "possession_score": 0.15,
        "defending_score": 0.20
    },

    "Defensive / Central Defender": {
        "attacking_score": 0.05,
        "creation_score": 0.05,
        "progression_score": 0.15,
        "possession_score": 0.25,
        "defending_score": 0.50
    }
}

#Calculate overall score
def calculate_performance_score(row):

    archetype = row["archetype"]

    weights = archetype_weights.get(
        archetype
    )

    if weights is None:
        return np.nan

    score = 0

    for category, weight in weights.items():

        score += (
            row[category]
            * weight
        )

    return score

