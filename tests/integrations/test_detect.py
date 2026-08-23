"""Which provider a client speaks to, and how far that answer goes."""

import anthropic
import openai
import pytest

from spillway.core.errors import ConfigurationError
from spillway.integrations import detect


def _anthropic(**kwargs):
    return anthropic.AsyncAnthropic(api_key="not-a-real-key", **kwargs)


def _openai(**kwargs):
    return openai.AsyncOpenAI(api_key="not-a-real-key", **kwargs)


@pytest.fixture(autouse=True)
def _forget_warnings():
    # The warning is once per host for the life of a process, which is right
    # for a process and wrong for a test suite that keeps making new clients.
    detect._warned_about_unofficial.clear()


def test_an_official_client_gets_its_own_adapter():
    assert detect.adapter_for(_anthropic()).name == "anthropic"
    assert detect.adapter_for(_openai()).name == "openai"


def test_the_same_protocol_elsewhere_gets_no_assumed_accounting():
    # The rule this module exists for. Correct protocol detection plus a
    # different host is a local engine or a hosted service, and applying the
    # named provider's rules there would be confidently wrong.
    local = _openai(base_url="http://localhost:8000/v1")
    assert detect.adapter_for(local).name == "openai_compatible"
    assert detect.adapter_for(local).charges_max_tokens() is False


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000/v1",
        "https://api.groq.com/openai/v1",
        "https://api.together.xyz/v1",
        "https://myproxy.internal/openai",
    ],
)
def test_every_unofficial_host_lands_on_the_generic_adapter(base_url):
    assert detect.adapter_for(_openai(base_url=base_url)).name == "openai_compatible"


def test_it_says_so_once_per_host(caplog):
    with caplog.at_level("WARNING"):
        detect.adapter_for(_openai(base_url="http://localhost:8000/v1"))
        detect.adapter_for(_openai(base_url="http://localhost:8000/v1"))
    assert len(caplog.records) == 1
    assert "not assuming" in caplog.records[0].getMessage()


def test_the_warning_names_the_way_to_say_it_really_is_the_provider(caplog):
    with caplog.at_level("WARNING"):
        detect.adapter_for(_openai(base_url="https://myproxy.internal/openai"))
    assert "provider='openai'" in caplog.records[0].getMessage()


def test_an_unrecognised_client_names_both_fixes():
    class NotAClient:
        pass

    with pytest.raises(ConfigurationError) as raised:
        detect.adapter_for(NotAClient())
    message = str(raised.value)
    assert "provider=" in message
    assert "spillway.admit()" in message


def test_the_host_is_read_off_the_client():
    assert detect.host_of(_anthropic()) == "api.anthropic.com"
    assert detect.host_of(_openai()) == "api.openai.com"
    assert detect.host_of(_openai(base_url="http://localhost:8000/v1")) == "localhost"


def test_a_port_does_not_confuse_the_host():
    # A base URL carries a port far more often than a provider's own does, and
    # comparing it as part of the host would send every one of them to the
    # generic adapter for the wrong reason.
    assert detect.host_of(_openai(base_url="https://api.openai.com:443/v1")) == "api.openai.com"


def test_asynchronous_and_synchronous_clients_are_told_apart():
    # By asking a method rather than by reading the class name, because a name
    # is a convention and this is a fact.
    adapter = detect.adapter_for(_anthropic())
    assert detect.is_asynchronous(_anthropic(), adapter) is True
    assert detect.is_asynchronous(anthropic.Anthropic(api_key="k"), adapter) is False

    adapter = detect.adapter_for(_openai())
    assert detect.is_asynchronous(_openai(), adapter) is True
    assert detect.is_asynchronous(openai.OpenAI(api_key="k"), adapter) is False


def test_an_endpoint_this_version_does_not_have_is_skipped():
    # An adapter names what a current library exposes. An older one has fewer,
    # and that should cost the caller nothing.
    class Sparse:
        base_url = "https://api.openai.com/v1"

    adapter = detect.adapter_for(_openai())
    assert detect.is_asynchronous(Sparse(), adapter) is True
