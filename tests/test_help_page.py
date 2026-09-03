"""Help quotes the model-license notices; it does not paraphrase them."""

from pathlib import Path

from minimax_studio.licenses import H3_TERRITORY, MUSIC_CREDIT
from minimax_studio.ui.pages.help_page import HELP
from minimax_studio.worker.catalog import H3_TERRITORY as CATALOG_H3


def _html() -> str:
    return HELP.format(
        version="0",
        music_credit=MUSIC_CREDIT,
        h3_territory=H3_TERRITORY,
    )


def test_help_quotes_music_credit_and_h3_territory() -> None:
    html = _html()
    assert MUSIC_CREDIT in html
    assert H3_TERRITORY in html
    assert "Apache-2.0" in html
    assert "from catalog" in html
    assert "you brought the file" in html


def test_catalog_reexports_the_same_h3_territory() -> None:
    assert CATALOG_H3 is H3_TERRITORY


def test_help_does_not_import_worker() -> None:
    text = Path("src/minimax_studio/ui/pages/help_page.py").read_text(encoding="utf-8")
    assert "minimax_studio.worker" not in text
    assert "from minimax_studio.licenses import" in text


def test_catalog_no_longer_owns_music_credit() -> None:
    text = Path("src/minimax_studio/worker/catalog.py").read_text(encoding="utf-8")
    assert "MUSIC_CREDIT" not in text
