# src/similarity.py
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity



midfielder_features = [
    #Scoring
    "goals_per90",
    "xg_per90",

    #Chance Creation
    "assists_per90",
    "xag_per90",
    "per_90_minutes_kp",
    "sca_sca90",

    #Progression
    "per_90_minutes_prgp",
    "per_90_minutes_prgc",
    "per_90_minutes_1_per_3",
    "per_90_minutes_ppa",


    #Possession/passing
    "pass_completion_pct",

    #Defense
    "per_90_minutes_tackles_tkl",
    "per_90_minutes_int"

]

forward_features = [
    # Scoring
    "goals_per90",
    "xg_per90",

    # Chance Creation
    "assists_per90",
    "xag_per90",
    "per_90_minutes_kp",
    "sca_sca90",

    # Progression / Attacking
    "per_90_minutes_prgc",
    "per_90_minutes_ppa",

    # Possession
    "pass_completion_pct",

    # Defensive Work
    "per_90_minutes_tackles_tkl",
    "per_90_minutes_int"
]

defender_features = [
    # Defensive Contribution
    "per_90_minutes_tackles_tkl",
    "per_90_minutes_int",

    # Passing
    "pass_completion_pct",

    # Progression
    "per_90_minutes_prgp",
    "per_90_minutes_prgc",
    "per_90_minutes_1_per_3",

    # Attacking Contribution
    "assists_per90",
    "xag_per90",
    "per_90_minutes_kp",

    # Optional scoring contribution
    "goals_per90",
    "xg_per90"
]

POSITION_FEATURES = {
    "Midfielder": midfielder_features,
    "Forward": forward_features,
    "Defender": defender_features
}

def prepare_similarity_data(
    dataframe,
    features
):

    X = dataframe[
        features
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(
        X
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_imputed
    )

    return X_scaled

def find_similar_players(
    player_name,
    dataframe,
    X_scaled,
    n=5,
    same_archetype=True,
    season=None
):

    # Find all rows for player
    matches = dataframe[
        dataframe["player"].str.lower()
        == player_name.lower()
    ]

    if matches.empty:
        return f"Player '{player_name}' not found."

    # If a season is requested, select that player's row
    if season is not None:

        season_matches = matches[
            matches["season"] == season
        ]

        if season_matches.empty:

            available_seasons = (
                matches["season"]
                .dropna()
                .sort_values()
                .unique()
            )

            return (
                f"Player '{player_name}' does not have data "
                f"for season {season}. "
                f"Available seasons: {available_seasons}"
            )

        matches = season_matches

    # Otherwise use most recent available season
    query_index = (
        matches
        .sort_values(
            "season",
            ascending=False
        )
        .index[0]
    )

    # Convert DataFrame index to matrix position
    query_position = (
        dataframe.index.get_loc(
            query_index
        )
    )

    query_vector = (
        X_scaled[
            query_position
        ]
        .reshape(1, -1)
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_vector,
        X_scaled
    )[0]

    results = dataframe.copy()

    results["similarity"] = (
        similarities
    )

    # Get query player's selected row
    query_player = dataframe.loc[
        query_index
    ]

    # Remove same player
    results = results[
        results["player"].str.lower()
        != player_name.lower()
    ]

    # If season specified, compare within same season
    if season is not None:

        results = results[
            results["season"] == season
        ]

    # Optional archetype restriction
    if same_archetype:

        results = results[
            results["archetype"]
            == query_player["archetype"]
        ]

    # Sort and remove duplicate players
    results = (
        results
        .sort_values(
            "similarity",
            ascending=False
        )
        .drop_duplicates(
            subset="player",
            keep="first"
        )
        .head(n)
    )

    return results[
        [
            "player",
            "team",
            "season",
            "player_positions",
            "archetype",
            "similarity"
        ]
    ]

def compare_players(
    player1,
    player2,
    position,
    position_data,
    position_features
):

    dataframe = position_data[position]
    features = position_features[position]

    p1_rows = dataframe[
        dataframe["player"].str.lower()
        == player1.lower()
    ]

    p2_rows = dataframe[
        dataframe["player"].str.lower()
        == player2.lower()
    ]

    if p1_rows.empty:
        raise ValueError(
            f"{player1} not found in {position} data."
        )

    if p2_rows.empty:
        raise ValueError(
            f"{player2} not found in {position} data."
        )

    p1 = (
        p1_rows
        .sort_values(
            "season",
            ascending=False
        )
        .iloc[0]
    )

    p2 = (
        p2_rows
        .sort_values(
            "season",
            ascending=False
        )
        .iloc[0]
    )

    comparison = pd.DataFrame({
        "Feature": features,

        player1: [
            p1[f]
            for f in features
        ],

        player2: [
            p2[f]
            for f in features
        ]
    })

    return comparison



