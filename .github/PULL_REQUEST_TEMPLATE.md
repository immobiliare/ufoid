## Description

Please include a concise summary, in clear English, of the changes in this pull request. If it closes an issue, please mention it here.

Closes: #(issue)

## 🎯 PRs Should Target Issues

Before your create a PR, please check to see if there is [an existing issue](https://github.com/immobiliare/ufoid/issues) for this change. If not, please create an issue before you create this PR, unless the fix is very small.

Not adhering to this guideline will result in the PR being closed.

## Tests

1. PRs will only be merged if tests pass on CI. To run the tests locally, please set up [your environment locally](https://github.com/immobiliare/vegeta/blob/main/CONTRIBUTING.md) and run the tests:`
   ```console
   source venv/bin/activate
   pytest --cov
   ```

2. You may need to run the linters:
   ```console
   source venv/bin/activate
   pre-commit run --all-files
   ```
