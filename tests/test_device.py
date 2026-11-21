import pytest
import torch

from emotion_classifier.device import pick_device

def _patch( monkeypatch, cuda: bool, mps: bool ) -> None:
    monkeypatch.setattr( torch.cuda, "is_available", lambda: cuda )
    monkeypatch.setattr( torch.backends.mps, "is_available", lambda: mps )

@pytest.mark.parametrize( "cuda, mps, expected", [
    ( True, True, "cuda" ),
    ( True, False, "cuda" ),
    ( False, True, "mps" ),
    ( False, False, "cpu" ),
] )
def test_pick_device_priority( monkeypatch, cuda, mps, expected ):
    _patch( monkeypatch, cuda, mps )
    assert pick_device() == expected

def test_pick_device_returns_a_known_device():
    assert pick_device() in ( "cuda", "mps", "cpu" )
