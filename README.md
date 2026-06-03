# How Far Can Prompting Go for Minimal-Edit Ukrainian Grammatical Error Correction?

Code for the paper: [How Far Can Prompting Go for Minimal-Edit Ukrainian Grammatical Error Correction?](https://unlp.org.ua/wp-content/uploads/2026/05/how-far-can-prompting-go-for-minimal-edit-ukrainian-grammatical-error-correction.pdf)

## Setup

```bash
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Run

Check `ua_validation_dir` in the config, then run:

```bash
uv run python main.py --config config.validation.yaml
uv run python main.py --config config.test.yaml
```

Outputs are saved to `outputs/<run_name>/`.
