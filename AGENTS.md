## Issue Tracking

This project uses **trx** for all issue tracking. Do NOT use markdown TODOs or other tracking methods.

```bash
trx ready --json                          # show unblocked work
trx create "title" -t bug -p 1 --json    # create issue (types: bug|feature|task|epic|chore, priority: 0-4)
trx update <id> --status in_progress      # claim work
trx close <id> --reason "Done"            # complete work
trx list --json                           # list issues
trx dep add <id> --blocks <other-id>      # manage dependencies
trx sync                                  # commit .trx/ with your code changes
```

Store AI planning docs in `history/`, not the repo root.
