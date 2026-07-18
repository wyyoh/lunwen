# Repository workflow

- Treat this repository as the source of truth for the Keyed-GRAM research project.
- After each completed modification, run the relevant validation, review the diff,
  create a focused Git commit, and push the current branch to `origin` before
  handing the work back to the user.
- Never commit datasets, virtual environments, access keys, generated private
  permutation keys, model checkpoints, optimizer state, or tensor/feature caches.
- Keep lightweight CSV/JSON experiment summaries and Markdown research reports
  versioned when they are needed to substantiate a conclusion.
- If validation or pushing is blocked, report the exact blocker instead of
  presenting the modification as remotely saved.
