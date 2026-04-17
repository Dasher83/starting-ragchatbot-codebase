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

## Dark/Light Theme Toggle

### Files Modified
- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

### Summary
Added a dark/light theme toggle button to the chat UI. The default theme remains dark; users can switch to light mode via a button in the top-right corner. The preference persists across page reloads via `localStorage`.

---

### `frontend/index.html`
- Added a `<button class="theme-toggle" id="themeToggle">` element immediately after `<body>`, positioned fixed top-right.
- Button contains two inline SVGs: a sun icon (shown in light mode) and a moon icon (shown in dark mode).
- Button has `aria-label="Toggle light/dark theme"` and each icon has `aria-hidden="true"` for accessibility.
- Bumped cache-busting version on `style.css` and `script.js` from `v=9` to `v=10`.

### `frontend/style.css`
- Added `[data-theme="light"]` CSS variable block after `:root` with light-mode color values:
  - `--background: #f1f5f9` (light slate)
  - `--surface: #ffffff`
  - `--surface-hover: #e2e8f0`
  - `--text-primary: #0f172a` (dark navy)
  - `--text-secondary: #64748b`
  - `--border-color: #cbd5e1`
  - `--assistant-message: #e2e8f0`
  - `--shadow` reduced opacity for light mode
  - `--welcome-bg: #dbeafe`
- Added a global `transition` rule (`background-color`, `border-color`, `color` — all 0.3s ease) for smooth theme switching.
- Added `.theme-toggle` styles: fixed position top-right, circular button, surface background, shadow, hover/focus states.
- Added `.icon-sun` / `.icon-moon` visibility rules using `opacity` + `transform` transitions — moon visible in dark mode, sun visible in light mode.

### `frontend/script.js`
- Added `initTheme()` function called on `DOMContentLoaded`:
  - Reads saved preference from `localStorage` and applies `data-theme="light"` to `document.documentElement` if needed.
  - Attaches click listener to `#themeToggle` that toggles the `data-theme` attribute and saves the new preference to `localStorage`.
