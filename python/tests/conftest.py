import pytest
from orthel.transforms import available_backends, use_backend


@pytest.fixture(autouse=True, params=available_backends())
def backend(request):
    """Run every test once per available transform backend.

    A module whose tests choose backends themselves overrides this fixture.
    """
    with use_backend(request.param):
        yield request.param
