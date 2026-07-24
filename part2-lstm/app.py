"""Flask web interface for the trained Part 2 PyTorch LSTM URL classifier.

The application analyzes the submitted URL text locally. It does not open,
resolve, or download the website.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from inference import load_predictor, predict_url

BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "artifacts" / "lstm_model.pt"
TEMPLATE_DIRECTORY = BASE_DIRECTORY / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIRECTORY))

# Load the model once when the server starts.
model, checkpoint, device = load_predictor(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    """Display the input form and classify one submitted URL string."""
    result = None
    error = None
    submitted_url = ""

    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()

        if not submitted_url:
            error = "Enter a URL string before running the analysis."
        elif len(submitted_url) > 10_000:
            error = "The submitted text is too long."
        else:
            result = predict_url(
                submitted_url,
                model=model,
                checkpoint=checkpoint,
                device=device,
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
        submitted_url=submitted_url,
        saved_metrics=checkpoint.get("metrics", {}),
        device=str(device),
        max_length=int(checkpoint.get("max_length", 0)),
        vocabulary_size=len(checkpoint.get("character_to_index", {})),
    )


if __name__ == "__main__":
    print("Part 2 LSTM web interface is ready at http://127.0.0.1:5001")
    # use_reloader=False prevents Flask from loading the PyTorch model twice.
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
