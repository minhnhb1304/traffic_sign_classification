# Contributing to Traffic Sign Classification

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to this project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by a standard Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

### Reporting Bugs

- Ensure the bug was not already reported by searching on GitHub under Issues.
- If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

- Open a new issue with a clear title and description.
- Explain why this enhancement would be useful to most users.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes (if applicable).
4. Update the documentation (e.g., `README.md`) if necessary.
5. Issue that pull request!

## Project Structure Guidelines

- **`api/`**: FastAPI backend code. Keep route handlers thin and move business/inference logic to services.
- **`frontend/`**: React/Vite frontend code. Follow standard React best practices (component modularity, hooks, etc.).
- **`src/`**: Machine Learning core pipeline. Ensure no UI or web API dependencies leak into this directory.

## Environment Setup

See the [README.md](README.md) for instructions on setting up the backend, frontend, and machine learning pipelines.
