"""UI smoke test: every sample article renders with a ground-truth comparison."""
# slider changes must keep the result visible (re-analysis without pressing Analyze again)
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted(p.stem for p in (ROOT / "samples").glob("*.txt"))


@pytest.mark.parametrize("sample", SAMPLES)
def test_sample_shows_truth(sample):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.selectbox[0].select(sample).run()
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Verdict" in labels and "Character accuracy" in labels
    assert any(m.value.startswith(("✅", "❌")) for m in at.markdown)


def test_slider_rerun_keeps_result():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.selectbox[0].select(SAMPLES[0]).run()
    at.button[0].click().run()
    at.sidebar.slider[1].set_value(0).run()
    assert not at.exception
    assert any(m.label == "Character accuracy" for m in at.metric)
