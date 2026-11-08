#!/bin/zsh
set -eu
ROOT_DIR="${0:A:h}"
APP_DIR="$ROOT_DIR/AITikiTaka/semiconductor-regression"
cd "$APP_DIR"
if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "먼저 설치.command를 실행해주세요."
  read -r "reply?Enter를 누르면 닫힙니다."
  exit 1
fi
echo "TikiTaka 공정 실험실을 시작합니다."
echo "브라우저에서 http://127.0.0.1:8502 을 열어주세요."
echo "종료하려면 이 창에서 Ctrl+C를 누르세요."
if curl -fsS --max-time 2 http://127.0.0.1:8502/_stcore/health >/dev/null 2>&1; then
  echo "이미 로컬 앱이 실행 중입니다."
  open http://127.0.0.1:8502
  exit 0
fi
mkdir -p local_data/logs
OLLAMA_BIN="$(command -v ollama || true)"
if [[ -z "$OLLAMA_BIN" && -x /opt/homebrew/bin/ollama ]]; then
  OLLAMA_BIN=/opt/homebrew/bin/ollama
fi
OLLAMA_PID=""
cleanup() {
  if [[ -n "$OLLAMA_PID" ]]; then kill "$OLLAMA_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
if [[ -n "$OLLAMA_BIN" ]] && ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  OLLAMA_NO_CLOUD=1 "$OLLAMA_BIN" serve >local_data/logs/ollama.log 2>&1 &
  OLLAMA_PID=$!
fi
"$ROOT_DIR/.venv/bin/python" -m streamlit run app_complete.py --server.address 127.0.0.1 --server.port 8502 --server.headless false --browser.gatherUsageStats false
