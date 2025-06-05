# automated prompt evaluation framework using OpenAI GPT and SentenceTransformers

This is a simple automated prompt evaluation framework using OpenAI GPT and SentenceTransformers.

## Files
- `prompts.csv`: Input prompts and expected responses.
- `evaluate_prompts.py`: Script to run the evaluation.
- `results.csv`: Auto-generated file with similarity scores.
- `requirements.txt`: Python dependencies.

## How to Run
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Add your OpenAI API key in `evaluate_prompts.py`.

3. Run the evaluation:
```
python evaluate_prompts.py
```

## Output
A `results.csv` file with prompt, response, and similarity score.
