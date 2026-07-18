from keyed_gram.bios import make_biographies


def test_biography_generator_is_deterministic():
    left = make_biographies(5, seed=3)
    right = make_biographies(5, seed=3)
    assert left == right
    assert len({record.name for record in left}) == 5
    assert len(left[0].evaluation_items()) == 4
