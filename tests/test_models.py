from music_stuff.models import ChordSegment, KeyEstimate


def test_key_estimate_label_combines_tonic_and_mode():
    key = KeyEstimate(tonic="C", mode="major", confidence=0.8)

    assert key.label == "C major"


def test_chord_segment_can_store_roman_numeral():
    chord = ChordSegment(symbol="Am", start=0.0, end=2.0, roman_numeral="vi")

    assert chord.symbol == "Am"
    assert chord.roman_numeral == "vi"
