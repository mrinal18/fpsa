"""Pinned upstream checkout, without altering tracked TRM source files."""
from pathlib import Path
import subprocess

TRM_URL = 'https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git'
TRM_SHA = 'c01103738605ba39d1430519b1ee0c62f4c707f8'
ROOT = Path(__file__).resolve().parents[2]


def checkout(path):
    path = Path(path).resolve()
    if not path.exists():
        subprocess.run(['git', 'clone', TRM_URL, str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', TRM_SHA], check=True)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != TRM_SHA:
        raise RuntimeError(f'Expected pinned TRM {TRM_SHA}; found {actual}. Use a fresh directory.')
    subprocess.run(['git', '-C', str(path), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    return path


def install_shims(path):
    """Only create our namespaced, untracked adapter/config files."""
    path = Path(path)
    shim = '# Generated FPSA ARC adapter; upstream tracked files remain unchanged.\n'
    shim += 'from src.fpsa_arc.trm_adapter import FPSAARC_ACTV1, ARCLossHead\n'
    dest = path / 'models' / 'fpsa_arc.py'
    if dest.exists() and not dest.read_text().startswith('# Generated FPSA ARC adapter'):
        raise FileExistsError(f'Refusing to overwrite {dest}')
    dest.write_text(shim)
    for config in (ROOT / 'configs' / 'arc').glob('fpsa_arc_*.yaml'):
        target = path / 'config' / 'arch' / config.name
        if target.exists() and not target.read_text().startswith('# Generated FPSA ARC config'):
            raise FileExistsError(f'Refusing to overwrite {target}')
        target.write_text(config.read_text())
