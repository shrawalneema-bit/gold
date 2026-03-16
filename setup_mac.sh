#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gold Price Predictor — One-command Mac setup
# Usage: bash setup_mac.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

GOLD="\033[33m"
GREEN="\033[32m"
RED="\033[31m"
DIM="\033[2m"
RESET="\033[0m"

echo ""
echo "${GOLD}🥇 Gold Price Predictor — Mac Setup${RESET}"
echo "${DIM}─────────────────────────────────────────${RESET}"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "${RED}❌ Python 3 not found.${RESET}"
    echo "   Install via: brew install python@3.11"
    echo "   Or:          pyenv install 3.11.9 && pyenv global 3.11.9"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PY_VERSION found"

if python3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
    :
else
    echo "${RED}⚠️  Python 3.10+ required. Current: $PY_VERSION${RESET}"
    exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "✓ Virtual environment exists"
fi
source .venv/bin/activate

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "📦 Installing dependencies (this may take ~2 min on first run)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✓ All packages installed"

# ── .env ─────────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env — add NEWS_API_KEY for extra news sources (optional)"
else
    echo "✓ .env exists"
fi

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p src/models/saved scripts

# ── Streamlit config ──────────────────────────────────────────────────────────
mkdir -p .streamlit
if [ ! -f ".streamlit/config.toml" ]; then
    cat > .streamlit/config.toml <<'EOF'
[theme]
primaryColor      = "#c9a84c"
backgroundColor   = "#0a0a0f"
secondaryBackgroundColor = "#0f0f1a"
textColor         = "#e8e8e8"
font              = "sans serif"

[server]
headless          = false
port              = 8501
enableXsrfProtection = false

[browser]
gatherUsageStats  = false
EOF
    echo "✓ Streamlit config created"
fi

echo ""
echo "${GREEN}✅ Setup complete!${RESET}"
echo ""
echo "  Next steps:"
echo ""
echo "  ${DIM}1. Activate the environment:${RESET}"
echo "     source .venv/bin/activate"
echo ""
echo "  ${DIM}2. (Recommended) Train the ML model locally with 5y data:${RESET}"
echo "     make train"
echo ""
echo "  ${DIM}3. Launch the dashboard:${RESET}"
echo "     make run"
echo "     → Opens at http://localhost:8501"
echo ""
echo "  ${DIM}4. After training, commit the model for Streamlit Cloud:${RESET}"
echo "     git add src/models/saved/ && git commit -m 'update trained model'"
echo ""
