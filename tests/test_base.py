def test_import_satwater():
    import satwater

    assert satwater is not None


def test_parse_shape():
    from satwater.utils import parse_shape

    assert parse_shape("128,128,3") == (128, 128, 3)


def test_parse_models_arg():
    from satwater.utils import parse_models_arg

    out = parse_models_arg("unet=a.keras,resnet34_256=b.keras")
    assert out["unet"] == "a.keras"
    assert out["resnet34_256"] == "b.keras"
