import os

from dotenv import load_dotenv
from openai import OpenAI

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# --------------------------------------------------
# Load OpenAI API key
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# Structured Output Models
# --------------------------------------------------

class ScoutingFactor(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    factor: Literal[
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
    ]

    direction: Literal[-1, 1]

    strength: float = Field(
        ge=0,
        le=1
    )

    confidence: float = Field(
        ge=0,
        le=1
    )

    evidence: str


class ScoutingExtraction(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    factors: list[ScoutingFactor]

    summary: str


# --------------------------------------------------
# LLM Instructions
# --------------------------------------------------

SCOUTING_SYSTEM_PROMPT = """
You are a soccer scouting information extraction system.

Your job is NOT to rate the player.

Your only job is to convert the user's qualitative scouting
observation into structured contextual factors.

Allowed factors:

leadership
- Ability to lead, organize, influence, or motivate teammates.

decision_making
- Quality of choices made during play.

work_rate
- Effort, intensity, willingness to contribute, and defensive effort.

tactical_fit
- How well the player fits the tactical system described by the user.

press_resistance
- Ability to operate effectively when pressured by opponents.

adaptability
- Ability to adjust to new positions, systems, leagues, or roles.

consistency
- Reliability of performance across games or situations.

injury_concern
- Availability, recurring injuries, or durability concerns.

communication
- Communication and organization with teammates.

off_ball_movement
- Movement, positioning, runs, and spatial awareness without the ball.

RULES:

1. Extract ONLY information supported by the user's text.

2. Do not invent characteristics that the user did not mention.

3. Do not use outside knowledge about the player.

4. Do not calculate a player rating or performance score.

5. Each factor may appear at most once.

6. Omit factors that are not mentioned or reasonably implied.

7. direction must be:
   +1 if the observation improves the player's evaluation.
   -1 if the observation lowers the player's evaluation.

8. strength represents how strongly the user describes the trait:
   0.0 = extremely weak statement
   1.0 = extremely strong statement.

9. confidence represents confidence in your interpretation of the
   user's text, not whether the user's statement is factually true.

10. evidence should briefly identify the wording that supports
    the extracted factor.

11. If the user's text contains no relevant scouting information,
    return an empty factors list.

Keep the interpretation conservative.
"""


# --------------------------------------------------
# LLM Factor Extraction
# --------------------------------------------------

def extract_factors_with_llm(
    user_text,
    model="gpt-6-luna"
):

    response = client.responses.parse(
        model=model,

        input=[
            {
                "role": "system",
                "content": SCOUTING_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_text
            }
        ],

        text_format=ScoutingExtraction
    )

    if response.output_parsed is None:

        raise ValueError(
            "The model did not return a valid structured response."
        )

    return (
        response.output_parsed
        .model_dump()
    )