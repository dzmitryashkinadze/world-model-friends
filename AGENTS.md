# Development Rules
- ASK QUESTIONS WHEN UNSURE
- ASK QUESTIONS WHEN DOING SOMETHING POTENTIALLY DANGEROUS
- ASK QUESTIONS WHEN DOING IMPORTANT TECH DECISIONS NOT SPECIFIED BY THE USER
- DO NOT LIST FILES IN data/ using `ls -R` as it has 5k files inside and it basically will flood your context

## Conversational Style
- Keep answers short and concise, no emojis, no fluff, technical prose only, be direct
- When the user asks a question, answer it first before doing anything else

## Code Quality
- Read files in full before wide-ranging changes

## Commands
- After code changes run pre-commit (not docs): `uv run pre-commit run -a` Fix all errors, warnings, and infos before committing.
- Never commit unless the user asks.
- Only commit files YOU changed in THIS session.
- Stage explicit paths (`git add <path1> <path2>`); never `git add -A` / `git add .`.
- Before committing, run `git status` and verify you are only staging your files.

# About the project
- the project uses `uv`
- project code is located in `world_model_friends` directory
- project tests are in `tests`
- you can use `rg` to quickly search relevant pieces for example  `rg --line-number --color always --smart-case -i -C 1 'import polars'`
- you can run test python code using `uv run python -c --% "print(\"Hello world\")"`
- use the web search functionality when you are not sure about the syntax / how it works
