# Repository Guidelines

## Project Structure & Module Organization

- `python/` contains the application: `main.py` starts the FastAPI/MJPEG service, while `Car.py` and `trafficLight.py` hold vehicle and traffic-light logic.
- `frontend/` contains the browser UI. The active application serves `frontend/main.html`; `toweb.py` is a separate, older streaming helper.
- `cord_data/points_data.json` stores stop-line and gate calibration entered in the UI. Treat its contents as runtime configuration.
- `inputVideos/` holds local source videos, `yaml/` holds model configuration, and `line_previews/` contains reference images.

## Build, Test, and Development Commands

Create an isolated environment and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the local service with `python python/main.py`; then open `http://127.0.0.1:8000`. Use `python python/main.py --device cpu` to force CPU inference, or `--model /path/to/model.pt` for a custom model. A model/video must be available as described in `README.md` before testing the full flow.

There is no automated test suite yet. At minimum, validate Python edits with `python -m py_compile python/*.py`, then exercise the affected API/UI path locally.

## Coding Style & Naming Conventions

Use four-space Python indentation, standard-library imports before third-party imports, and `snake_case` for functions, variables, and modules. Use `PascalCase` for classes (for example, `TrafficLightDetector`) and `UPPER_SNAKE_CASE` for application constants. Keep comments and user-facing strings consistent with the surrounding Traditional Chinese. No formatter or linter is configured; preserve nearby style and keep changes focused.

## Testing Guidelines

Place future tests under `tests/` using `test_<feature>.py` filenames and `test_<behavior>` functions. Prefer unit tests for calculation and detection helpers; use small fixture data rather than video files or model downloads. Add a regression test when fixing a behavior defect.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive summaries, often in Traditional Chinese (for example, `frontend` or `加入車輛行駛狀態與鏡頭運動補償`). Write an imperative, focused subject; include a scope when useful, such as `frontend: 修正 Gate 儲存`. Keep unrelated generated files out of commits.

Pull requests should explain the behavior change, list validation performed, link related issues when available, and include screenshots for UI or annotated-video changes. Call out model, input-video, or calibration-data requirements explicitly.
