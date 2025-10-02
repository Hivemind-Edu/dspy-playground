import dspy
import os
import base64


from openinference.instrumentation.dspy import DSPyInstrumentor

from langfuse import get_client
import dotenv


import json

dotenv.load_dotenv()

DSPyInstrumentor().instrument()

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


with open("annotated-results-only-score.json", "r") as f:
    annotated_results = json.load(f)


examples = [
    (
        dspy.Example(
            feed=item.get("blueprint"),
            score=item.get("userScore"),
            feedback=item.get("userFeedback"),
        ).with_inputs("feed")
    )
    for item in annotated_results
]

pro = dspy.LM(
    "gemini/gemini-2.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    # both values are from 0 to 1. the closer they match the better
    score = 1.0 - abs(example.score - prediction.score)

    # example.feedback and prediction.feedback
    fb = f"""This is the feedback from the gold dataset: 
    <GOLD_FEEDBACK>
    {example.feedback}
    <GOLD_FEEDBACK>
    This is the feedback that you gave. Please compare and adjust.
    <PREDICTION_FEEDBACK>
    {prediction.feedback}
    <PREDICTION_FEEDBACK>
    """

    return dspy.Prediction(score=score, feedback=fb)


dspy.configure(lm=pro)

# Initialize optimizer
optimizer = dspy.GEPA(
    metric=metric,
    auto="light",
    reflection_lm=pro,
)


signature = dspy.Signature(
    "feed -> score: float, feedback: str",
    instructions="""The output should be evaluated with a score from 0 to 1. The closer the score is to 1, the better the output is.

    Evaluation Criteria:

    - engaging: The content is engaging (not boring)
    - educational: The content is educational (not shallow): After reading the content, the user should have a deep understanding of the topic.
    - Language Variety: The content uses different language styles (not always the same)
    - Non-AI-like: The content doesn't seem AI-generated. Examples of AI-like text:
      - em-dashes
      - overly smooth and generic phrasing like “In today’s fast-paced world, it is crucial to…”
      - overuse of “however,” “on the other hand,” “it is important to note.”
      - precise but non-technical synonyms (e.g., “significant,” “notable,” “crucial”
      - Few “I” statements, anecdotes, or subjective judgments unless prompted.
    - Display Style Distribution: The distribution of post display styles is balanced, and the correct display styles are used where it makes sense.
    """,
)


program = dspy.Predict(signature=signature, lm=pro)


# Optimize program
print("Optimizing program with GEPA...")
optimized_program = optimizer.compile(
    program,
    trainset=examples,
)

# Save optimize program for future use
optimized_program.save("optimized.json")
