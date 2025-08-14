import json
import pytest


def pytest_addoption(parser):
    parser.addoption("--config", action="store")


@pytest.fixture(scope="session")
def cdk_outputs(pytestconfig):
    path = pytestconfig.getoption("config")
    if not path:
        pytest.skip("Argument required: --config")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def api_url(cdk_outputs):

    base_url = cdk_outputs["ApiStack"]["ApiGetPredictionUrl"]
    resource = cdk_outputs["ApiStack"]["ApiGetPredictionResourcePredict"]

    return f"{base_url}{resource}"


@pytest.fixture(scope="session")
def lambda_name_call_end(cdk_outputs):
    return cdk_outputs["LambdaStack"]["EndpointLambdaFuncName"]
