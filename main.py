import dspy
import os
import dotenv
import base64

from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

# Import the optimizer
import yaml

DSPyInstrumentor().instrument()

dotenv.load_dotenv()

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


with open("dataset/data.yaml", "r") as f:
    dataset_raw = yaml.safe_load(f)


examples = [
    (
        dspy.Example(
            post=item.get("post"),
            image=dspy.Image.from_file(f"dataset/{item.get('image')}"),
            result=item.get("result"),
            explanation=item.get("explanation"),
        ).with_inputs("post", "image")
    )
    for item in dataset_raw
]


# Initialize the LM
flash = dspy.LM(
    "gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    reasoning_effort=None,
)

pro = dspy.LM(
    "gemini/gemini-2.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    gold = example.result or []
    pred = getattr(prediction, "result", None) or []
    # normalize types
    if isinstance(pred, tuple):
        pred = list(pred)
    is_correct = gold == pred
    score = 1.0 if is_correct else 0.0

    if is_correct:
        fb = f"Correct. Kept constraints. Reasoning: {example.explanation}"
    else:
        fb = (
            f"Incorrect.\nExpected: {gold}\nGot: {pred}\n"
            f"Reasoning to follow: {example.explanation}"
        )
        if pred_name is not None:
            fb = f"[{pred_name}] {fb}"
    return dspy.Prediction(score=score, feedback=fb)


class MySignature(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    post: str = dspy.InputField()
    result: list[int] = dspy.OutputField()


dspy.configure(lm=flash)

program = dspy.ChainOfThought(signature=MySignature, lm=flash)

# Initialize optimizer
optimizer = dspy.GEPA(
    metric=metric,
    auto="light",
    reflection_lm=pro,
    candidate_selection_strategy="current_best",
    instruction_proposer=MultiModalInstructionProposer(),
)

# Optimize program
print("Optimizing program with GEPA...")
optimized_program = optimizer.compile(
    program,
    trainset=examples,
)

# Save optimize program for future use
optimized_program.save("optimized.json")
