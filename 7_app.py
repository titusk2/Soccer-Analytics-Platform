import streamlit as st
import pandas as pd

from src.scoring import get_player_score

from src.similarity import (
    prepare_similarity_data,
    find_similar_players,
    compare_players,
    POSITION_FEATURES
)

from src.ai_extraction import extract_factors_with_llm
from src.personalization import evaluate_with_scouting_note


st.set_page_config(
    page_title="Soccer Analytics Platform",
    layout="wide"
)


midfielders = pd.read_csv(
    "data/processed/ midfielders_scored.csv"
)

forwards = pd.read_csv(
    "data/processed/ forwards_scored.csv"
)

defenders = pd.read_csv(
    "data/processed/ defenders_scored.csv"
)

position_data = {
    "Midfielder": midfielders,
    "Forward": forwards,
    "Defender": defenders
}


selected_position = st.selectbox(
    "Select Position Group",
    [
        "Midfielder",
        "Forward",
        "Defender"
    ]
)

df = position_data[
    selected_position
]


st.title("Soccer Analytics Platform")


player_list = sorted(
    df["player"]
    .dropna()
    .unique()
)


selected_player = st.selectbox(
    "Select a player",
    player_list
)

player_rows = df[
    df["player"] == selected_player
]

available_seasons = sorted(
    player_rows["season"]
    .dropna()
    .unique(),
    reverse=True
)

selected_season = st.selectbox(
    "Select Season",
    available_seasons
)


player = get_player_score(
    selected_player,
    selected_position,
    position_data,
    season=selected_season
)


st.subheader(selected_player)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Performance Score",
        round(
            player["performance_score"],
            1
        )
    )

with col2:
    st.metric(
        "Performance Percentile",
        f"{player['performance_percentile']:.1f}%"
    )

with col3:
    st.metric(
        "Data Completeness",
        f"{player['data_completeness']:.0f}%"
    )


st.write(
    "**Team:**",
    player["team"]
)

st.write(
    "**Season:**",
    player["season"]
)

st.write(
    "**Archetype:**",
    player["archetype"]
)

st.subheader("Performance Profile")

profile_cols = [
    "attacking_score",
    "creation_score",
    "progression_score",
    "possession_score",
    "defending_score"
]

profile_labels = {
    "attacking_score": "Attacking",
    "creation_score": "Chance Creation",
    "progression_score": "Progression",
    "possession_score": "Possession",
    "defending_score": "Defending"
}

profile_df = pd.DataFrame({
    "Category": [
        profile_labels[col]
        for col in profile_cols
    ],
    "Score": [
        player[col]
        for col in profile_cols
    ]
})

st.bar_chart(
    profile_df.set_index("Category")
)

df = position_data[
    selected_position
]

features = POSITION_FEATURES[
    selected_position
]

X_scaled = prepare_similarity_data(
    df,
    features
)

@st.cache_data
def prepare_similarity_matrix(
    dataframe,
    features
):

    return prepare_similarity_data(
        dataframe,
        features
    )


X_scaled = prepare_similarity_matrix(
    df,
    features)


# -------------------------
# Performance Profile
# -------------------------

st.subheader("Performance Profile")

profile_cols = [
    "attacking_score",
    "creation_score",
    "progression_score",
    "possession_score",
    "defending_score"
]

profile_labels = {
    "attacking_score": "Attacking",
    "creation_score": "Chance Creation",
    "progression_score": "Progression",
    "possession_score": "Possession",
    "defending_score": "Defending"
}

profile_df = pd.DataFrame({
    "Category": [
        profile_labels[col]
        for col in profile_cols
    ],
    "Score": [
        player[col]
        for col in profile_cols
    ]
})

st.bar_chart(
    profile_df.set_index("Category")
)


# -------------------------
# Similar Players
# -------------------------

feature_labels = {
    "goals_per90": "Goals / 90",
    "xg_per90": "xG / 90",
    "assists_per90": "Assists / 90",
    "xag_per90": "xAG / 90",
    "per_90_minutes_kp": "Key Passes / 90",
    "sca_sca90": "Shot-Creating Actions / 90",
    "per_90_minutes_prgp": "Progressive Passes / 90",
    "per_90_minutes_prgc": "Progressive Carries / 90",
    "per_90_minutes_1_per_3": "Final Third Entries / 90",
    "per_90_minutes_ppa": "Passes Into Penalty Area / 90",
    "pass_completion_pct": "Pass Completion %",
    "per_90_minutes_tackles_tkl": "Tackles / 90",
    "per_90_minutes_int": "Interceptions / 90"
}

st.divider()

st.subheader("Similar Players")

num_similar = st.slider(
    "Number of similar players",
    3,
    10,
    5
)

same_archetype = st.checkbox(
    "Only show players from the same archetype",
    value=True
)

similar_players = find_similar_players(
    selected_player,
    df,
    X_scaled,
    n=num_similar,
    same_archetype=same_archetype
)

if isinstance(
    similar_players,
    pd.DataFrame
):

    similar_players = (
        similar_players.copy()
    )

    similar_players["Similarity %"] = (
        similar_players["similarity"]
        * 100
    ).round(1)

    st.dataframe(
        similar_players[
            [
                "player",
                "team",
                "season",
                "player_positions",
                "archetype",
                "Similarity %"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        similar_players
    )

st.divider()

st.subheader("Player Comparison")

col1, col2 = st.columns(2)

with col1:
    player1 = st.selectbox(
        "Player 1",
        player_list,
        index=0,
        key="comparison_player1"
    )

with col2:
    player2 = st.selectbox(
        "Player 2",
        player_list,
        index=1,
        key="comparison_player2"
    )


if player1 != player2:

    comparison = compare_players(
        player1,
        player2,
        selected_position,
        position_data,
        POSITION_FEATURES
    )

    comparison_display = comparison.copy()

    # Replace raw feature names with readable labels
    comparison_display["Feature"] = (
        comparison_display["Feature"]
        .replace(feature_labels)
    )

    # Round statistics
    comparison_display[player1] = (
        comparison_display[player1]
        .round(2)
    )

    comparison_display[player2] = (
        comparison_display[player2]
        .round(2)
    )

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn(
                "Metric",
                width="large"
            ),

            player1: st.column_config.NumberColumn(
                player1,
                format="%.2f"
            ),

            player2: st.column_config.NumberColumn(
                player2,
                format="%.2f"
            )
        }
    )

else:

    st.info(
        "Select two different players to compare."
    )

st.markdown("### Category Comparison")
player1_data = (
    df[
        df["player"].str.lower()
        == player1.lower()
    ]
    .sort_values(
        "season",
        ascending=False
    )
    .iloc[0]
)

player2_data = (
    df[
        df["player"].str.lower()
        == player2.lower()
    ]
    .sort_values(
        "season",
        ascending=False
    )
    .iloc[0]
)

category_labels = {
    "attacking_score": "Attacking",
    "creation_score": "Chance Creation",
    "progression_score": "Progression",
    "possession_score": "Possession",
    "defending_score": "Defending"
}

category_comparison = pd.DataFrame({
    "Category": list(
        category_labels.values()
    ),

    player1: [
        player1_data[col]
        for col in category_labels
    ],

    player2: [
        player2_data[col]
        for col in category_labels
    ]
})

category_comparison[player1] = (
    category_comparison[player1]
    .round(1)
)

category_comparison[player2] = (
    category_comparison[player2]
    .round(1)
)

st.dataframe(
    category_comparison,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Category": st.column_config.TextColumn(
            "Category"
        ),

        player1: st.column_config.NumberColumn(
            player1,
            format="%.1f"
        ),

        player2: st.column_config.NumberColumn(
            player2,
            format="%.1f"
        )
    }
)

category_comparison = pd.DataFrame({
    "Category": [
        "Overall Performance",
        "Attacking",
        "Chance Creation",
        "Progression",
        "Possession",
        "Defending"
    ],

    player1: [
        player1_data["performance_score"],
        player1_data["attacking_score"],
        player1_data["creation_score"],
        player1_data["progression_score"],
        player1_data["possession_score"],
        player1_data["defending_score"]
    ],

    player2: [
        player2_data["performance_score"],
        player2_data["attacking_score"],
        player2_data["creation_score"],
        player2_data["progression_score"],
        player2_data["possession_score"],
        player2_data["defending_score"]
    ]
})

chart_df = (
    category_comparison[
        category_comparison["Category"]
        != "Overall Performance"
    ]
    .set_index("Category")
)

st.bar_chart(chart_df)

st.divider()

st.subheader("AI Scout")

st.write(
    "Add your own scouting observations to generate "
    "a personalized evaluation."
)

scouting_note = st.text_area(
    "Scouting observations",
    placeholder=(
        "Example: He is an excellent decision maker and "
        "communicates well, but struggles against aggressive pressing."
    ),
    height=150,
    key="scouting_note"
)

analyze_button = st.button(
    "Analyze Scouting Note",
    key="analyze_scouting_note"
)

if analyze_button:

    if not scouting_note.strip():

        st.warning(
            "Enter a scouting observation first."
        )

    else:

        with st.spinner(
            "Analyzing scouting observation..."
        ):

            try:

                personalized_result = evaluate_with_scouting_note(
                    selected_player,
                    selected_position,
                    scouting_note,
                    position_data,
                    season=selected_season
                )

                st.subheader(
                    "Personalized Evaluation"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Baseline Score",
                        f"{personalized_result['base_score']:.2f}"
                    )

                with col2:
                    st.metric(
                        "Personalized Score",
                        f"{personalized_result['personalized_score']:.2f}"
                    )

                with col3:
                    st.metric(
                        "Score Adjustment",
                        f"{personalized_result['total_adjustment']:+.2f}"
                    )

                # Outside metric columns
                st.markdown(
                    "### Contextual Factors"
                )

                factors = personalized_result[
                    "factors"
                ]

                if factors:

                    factor_rows = []

                    for factor in factors:

                        factor_rows.append({
                            "Factor": (
                                factor["factor"]
                                .replace("_", " ")
                                .title()
                            ),

                            "Strength": round(
                                factor["strength"],
                                2
                            ),

                            "Confidence": round(
                                factor["confidence"],
                                2
                            ),

                            "Adjustment": round(
                                factor["adjustment"],
                                2
                            )
                        })

                    factor_df = pd.DataFrame(
                        factor_rows
                    )

                    st.dataframe(
                        factor_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No relevant contextual factors "
                        "were identified."
                    )

            except Exception as e:

                st.error(
                    f"Unable to analyze scouting note: {e}"
                )