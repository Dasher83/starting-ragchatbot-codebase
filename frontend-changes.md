# Frontend Changes

## Code Quality Tooling

Added Prettier and ESLint to the frontend development workflow for consistent code formatting and static analysis.

### New Files

- `frontend/package.json` — npm project config with `format`, `format:check`, `lint`, and `quality` scripts
- `frontend/.prettierrc` — Prettier config (100-char print width, single quotes, ES5 trailing commas, 2-space indent)
- `frontend/eslint.config.js` — ESLint flat config for browser globals and rules (no `var`, `===` enforcement, `no-undef`)

### Modified Files

- `frontend/script.js` — Reformatted by Prettier; changed `catch (e)` to `catch` to remove unused-variable warning
- `frontend/index.html` — Reformatted by Prettier (consistent attribute quoting and indentation)
- `frontend/style.css` — Reformatted by Prettier (consistent spacing and property ordering style)

### Development Scripts

Run from `frontend/` after `npm install`:

| Command | Purpose |
|---|---|
| `npm run format` | Auto-format all files with Prettier |
| `npm run format:check` | Check formatting without writing (CI-safe) |
| `npm run lint` | Run ESLint on `script.js` |
| `npm run quality` | Run both format check and lint |
