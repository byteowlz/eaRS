## Issue Tracking with trx

**IMPORTANT**: This project uses **trx** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why trx?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: CRDT-based storage in `.trx/` for version control
- Agent-optimized: JSON output, ready work detection, parent/child relationships
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
trx ready --json
```

**Create new issues:**
```bash
trx create "Issue title" -t bug|feature|task -p 0-4 --json
trx create "Child issue" -t task -p 1 --parent trx-123 --json
```

**Claim and update:**
```bash
trx update trx-42 --status in_progress --json
trx update trx-42 -p 1 --json
```

**Complete work:**
```bash
trx close trx-42 --reason "Completed" --json
```

**Manage dependencies:**
```bash
trx dep add trx-42 trx-10       # trx-42 depends on trx-10
trx dep rm trx-42 trx-10        # remove dependency
trx dep tree --json              # show dependency tree
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `trx ready` shows unblocked issues
2. **Claim your task**: `trx update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `trx create "Found bug" -t bug -p 1 --parent <parent-id>`
5. **Complete**: `trx close <id> --reason "Done"`
6. **Sync**: Run `trx sync` to commit `.trx/` changes, or commit the `.trx/` directory together with your code changes so issue state stays in sync with code state

### Managing AI-Generated Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

**Best Practice: Use a dedicated directory for these ephemeral files**

**Recommended approach:**
- Create a `history/` directory in the project root
- Store ALL AI-generated planning/design docs in `history/`
- Keep the repository root clean and focused on permanent project files
- Only access `history/` when explicitly asked to review past planning

**Example .gitignore entry (optional):**
```
# AI planning documents (ephemeral)
history/
```

### Important Rules

- Use trx for ALL task tracking
- Always use `--json` flag for programmatic trx commands
- Link discovered work with `--parent` to connect related issues
- Check `trx ready` before asking "what should I work on?"
- Store AI planning docs in `history/` directory
- Do NOT create markdown TODO lists
- Do NOT use external issue trackers
- Do NOT duplicate tracking systems
- Do NOT clutter repo root with planning documents
