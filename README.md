# Breakout Radar

Breakout Radar is a Streamlit dashboard for scanning Indian equities for technical breakout setups across major NIFTY universes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Use `app.py` as the entrypoint file.
4. If prompted, keep the app root at the repository root and choose a supported Python version.

## Files

- `app.py` - Streamlit entrypoint
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - shared app theme for local and cloud runs
