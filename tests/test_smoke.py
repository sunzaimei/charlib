"""End-to-end smoke test: CHARLIB_TEST_GLB=path/to.glb uv run pytest tests -q  (or: python -m pytest tests -q, or run this file directly)."""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_build_end_to_end():
    glb = os.environ.get('CHARLIB_TEST_GLB')
    if not glb:
        import pytest; pytest.skip('set CHARLIB_TEST_GLB')
    from charlib.build import build
    from charlib.config import Config
    out = tempfile.mkdtemp()
    rep = build(glb, out, Config(name='Test', previews=False))
    assert os.path.exists(f'{out}/Test_library.blend') and os.path.exists(f'{out}/Test_library.glb')
    assert len(rep['actions']) >= 22
    for e in Config().expressions:
        assert os.path.exists(f'{out}/textures/face_{e}.jpg'), e
    assert json.load(open(f'{out}/build_report.json'))['landmarks']['source'] in ('auto', 'config')


if __name__ == '__main__':
    test_build_end_to_end(); print('ok')
