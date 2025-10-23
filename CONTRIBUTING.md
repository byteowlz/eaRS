## Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/) with automated enforcement via git hooks.

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: New feature (triggers minor version bump)
- **fix**: Bug fix (triggers patch version bump)
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements (triggers patch version bump)
- **test**: Adding or updating tests
- **build**: Build system changes
- **ci**: CI/CD changes
- **chore**: Other changes
- **revert**: Revert a previous commit

### Breaking Changes

Add `!` after type/scope or add `BREAKING CHANGE:` in footer to trigger major version bump:

```
feat!: remove deprecated API
```

or

```
feat: new authentication system

BREAKING CHANGE: old auth tokens are no longer supported
```

### Examples

```bash
feat: add WebSocket support for live transcription
fix: resolve memory leak in audio buffer
docs: update installation instructions
perf: optimize Whisper model loading
```

## Version Bumping

### Using cargo-release (manual)

```bash
cargo release patch   # 0.3.3 -> 0.3.4
cargo release minor   # 0.3.3 -> 0.4.0  
cargo release major   # 0.3.3 -> 1.0.0
```

This will NOT publish to crates.io (configured in `release.toml`).

### Using semantic-release (automated)

Push to main branch and GitHub Actions will automatically:
1. Analyze commits since last release
2. Determine version bump based on commit types
3. Update Cargo.toml
4. Generate CHANGELOG.md
5. Create git tag
6. Create GitHub release

Version bumps follow this logic:
- `feat:` commits -> minor bump
- `fix:` or `perf:` commits -> patch bump
- `BREAKING CHANGE:` or `!` -> major bump
