"""Guard against exposing internal implementation milestones in the repository."""
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_LABEL = re.compile(r'\b(?:stage|phase)[\s_-]*\d+\b', re.IGNORECASE)
TEXT_EXTENSIONS = {'.md', '.txt', '.rst', '.py', '.toml', '.yaml', '.yml', '.json', '.sh'}
IGNORED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache',
           'var', 'artifacts', 'demo_uploads', 'data', 'node_modules',
           'mlruns', 'mlartifacts', 'cloud_exports'}


def test_repository_has_no_internal_milestone_labels():
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT,
                            capture_output=True, check=True)
    paths = [Path(p.decode()) for p in result.stdout.split(b'\0') if p]
    matches = []
    for path in paths:
        if path.as_posix() == 'tests/test_repository_presentation.py':
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS or any(p in IGNORED for p in path.parts):
            continue
        file = ROOT / path
        if not file.is_file():
            continue
        for line_number, line in enumerate(file.read_text(encoding='utf-8', errors='replace').splitlines(), start=1):
            if INTERNAL_LABEL.search(line):
                matches.append(f'{path}:{line_number}')
    assert not matches, 'Internal implementation labels remain in: ' + ', '.join(matches)
