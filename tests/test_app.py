from streamlit.testing.v1 import AppTest


def test_app_loads_filters_and_predicts():
    app = AppTest.from_file("../streamlit_app.py", default_timeout=30).run()
    assert not app.exception
    bedrooms = app.slider(key="bedroom_above_ground")
    assert bedrooms.label == "🛏️ Bedrooms above ground"
    assert bedrooms.min == 0
    assert bedrooms.max == 8
    assert bedrooms.value == 3
    bedrooms.set_value(4).run(timeout=30)
    assert not app.exception
    assert app.slider(key="bedroom_above_ground").value == 4
    app.selectbox[0].select("👑 Luxury — above $400K").run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.success
    assert "Estimated Sale Price" in app.success[0].value
