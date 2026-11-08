#!/bin/zsh
set -eu
ROOT_DIR="${0:A:h}"
cd "$ROOT_DIR"
echo "프로젝트 폴더에 전용 Python 환경을 설치합니다. 최초 설치에는 인터넷이 필요합니다."
if [[ ! -x .venv/bin/python ]]; then
  PYTHON_BIN=""
  for candidate in python3.12 python3.11 python3.10 python3.9 python3 /usr/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3, 9) <= sys.version_info[:2] < (3, 13)))' 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3.9~3.12를 먼저 설치해주세요. 예: brew install python@3.12"
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r AITikiTaka/semiconductor-regression/requirements-app.txt
if [[ "$(uname -s)" == Darwin && ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]]; then
  if [[ -x /opt/homebrew/bin/brew ]]; then
    /opt/homebrew/bin/brew install libomp
  else
    echo "XGBoost 실행에는 OpenMP가 필요합니다. Homebrew 설치 후 brew install libomp를 실행해주세요."
  fi
fi
echo "설치 완료. 실행.command를 더블클릭해주세요."
read -r "reply?Enter를 누르면 닫힙니다."
