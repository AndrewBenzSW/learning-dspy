from flask import Flask, render_template, request

# Import our agent function
from writing_agent import transform_text

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index_get():
    """
    Handle both displaying the form (GET) and processing submissions (POST).

    This is a classic "form posts back to itself" pattern:
    - GET: Show empty form
    - POST: Process the input, show form again with results
    """
    result = None
    original_text = ""
    task = ""

    # Render the template with current state
    # On GET: all values are empty/None
    # On POST: values are populated so user sees what they submitted
    return render_template(
        "index.html",
        result=result,
        original_text=original_text,
        task=task
    )


@app.route("/", methods=["POST"])
def index_post():
  # Get form data
  original_text = request.form.get("text", "")
  task = request.form.get("task", "")

  if original_text and task:
      # Call our DSPy agent (this is synchronous - browser waits)
      result = transform_text(original_text, task)


if __name__ == "__main__":
    # debug=True enables auto-reload when you edit code
    # host="0.0.0.0" makes it accessible from other machines (optional)
    app.run(debug=True, port=5000)
