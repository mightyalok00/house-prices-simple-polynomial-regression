from streamlit.testing.v1 import AppTest


def test_app_loads_filters_and_predicts():
    app = AppTest.from_file("../app.py", default_timeout=30).run()
    assert not app.exception
    app.selectbox[0].select("👑 Luxury — above $400K").run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.success
    assert "Estimated Sale Price" in app.success[0].value
