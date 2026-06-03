# Development Workflow

## Code Quality Standards

To maintain high code quality and consistency, all changes must adhere to the following rules:

### 1. Linting and Formatting
- Use `ruff` for all linting and formatting tasks.
- **Mandatory Step**: After every code modification, the agent **must** run the following commands to verify compliance:
  ```bash
  uv run ruff check .
  uv run ruff format --check .
  ```
  If these commands fail, the agent must fix the issues and re-run the commands before considering the task complete.

### 2. Dependency Management
- All dependencies should be managed using `uv`.
- Update `pyproject.toml` when adding new libraries.

### 3. Commit Standards
- Use `pre-commit` to ensure code quality before committing.
