from pathlib import Path
from src.config import FEATURE_COLUMNS, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
FEATURES = FEATURE_COLUMNS
TARGETS = TARGET_COLUMNS
FIELDS = FEATURES + TARGETS
LABELS = dict(zip(FIELDS, ['RF 전력', '챔버 압력', '공정 온도', '가스 유량', '박막 두께', '균일도']))
UNITS = dict(zip(FIELDS, ['W', 'mTorr', '°C', 'sccm', 'nm', '%']))
UNIT_OPTIONS = {
    'RF_Power': ['W', 'kW'], 'Pressure': ['mTorr', 'Torr', 'Pa'],
    'Temperature': ['°C', 'K'], 'Gas_Flow_Rate': ['sccm', 'slm'],
    'Thickness': ['nm', 'µm'], 'Uniformity': ['%', 'fraction'],
}
ALIASES = {
    'RF_Power': ['rfpower', 'rf전력', '인가전력', 'rf출력'],
    'Pressure': ['pressure', '챔버압력', '공정압력'],
    'Temperature': ['temperature', '공정온도', '설정온도', 'tempc'],
    'Gas_Flow_Rate': ['gasflowrate', 'gasflow', '가스유량'],
    'Thickness': ['thickness', '측정두께', '박막두께'],
    'Uniformity': ['uniformity', '균일도'],
}
