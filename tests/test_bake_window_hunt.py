"""Bake-window adversarial bug hunt (round 1).

Targets the @beta surface shipped 2026-06-10/12 that the v1.0 plan bakes for
three weeks: ``selectools.a2a`` JSON-RPC validation and ``selectools.pending``
confirm parsing. Each test is an adversarial repro that failed against the
code as shipped; the fixes are committed alongside.
"""

from __future__ import annotations

import pytest

from selectools.a2a.server import A2AServer, _extract_prompt
from selectools.a2a.types import INVALID_PARAMS
from selectools.pending import RegexConfirmParser
from selectools.types import AgentResult, Message, Role

# ---------------------------------------------------------------------------
# Finding 1 — A2A: a message.parts list that passes validation (one text
# dict) but carries a non-dict element crashed _extract_prompt with an
# unhandled AttributeError, surfacing as HTTP 500 instead of a clean
# JSON-RPC -32602 Invalid params.
# ---------------------------------------------------------------------------


class _FakeAgent:
    class config:  # noqa: D106 - test stub
        name = "t"

    tools: list = []

    def clone_for_isolation(self) -> "_FakeAgent":
        return self

    def run(self, messages: object) -> AgentResult:
        return AgentResult(message=Message(role=Role.ASSISTANT, content="ok"), iterations=1)


def _testclient(server: A2AServer):
    from starlette.testclient import TestClient

    return TestClient(server, raise_server_exceptions=False)


class TestA2ANonDictPart:
    def test_extract_prompt_tolerates_non_dict_parts(self) -> None:
        # Must not raise AttributeError on a stray non-dict list element.
        out = _extract_prompt([{"kind": "text", "text": "hi"}, "garbage"])
        assert out == "hi"

    def test_message_send_non_dict_part_is_invalid_params_not_500(self) -> None:
        server = A2AServer(agent=_FakeAgent())
        client = _testclient(server)
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {"message": {"parts": [{"kind": "text", "text": "hi"}, "garbage"]}},
        }
        resp = client.post("/a2a", json=body)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert "error" in payload, payload
        assert payload["error"]["code"] == INVALID_PARAMS


# ---------------------------------------------------------------------------
# Finding 2 — pending: RegexConfirmParser fired a destructive CONFIRM on a
# message that restates a destructive verb behind a NON-leading negation
# ("se você não pode apagar" / "tú no puedes borrar"). The module's stated
# bias is conservative (a false positive fires a destructive action), so a
# negated restatement must never confirm.
# ---------------------------------------------------------------------------


class TestParserMidSentenceNegation:
    @pytest.fixture
    def parser(self) -> RegexConfirmParser:
        return RegexConfirmParser()

    @pytest.mark.parametrize(
        "msg",
        [
            "se você não pode apagar, tudo bem",
            "espero que não pode apagar isso",
            "acho que você não pode deletar",
            "tú no puedes borrar",
            "creo que no puedes eliminar",
        ],
    )
    def test_negated_restatement_never_confirms(self, parser: RegexConfirmParser, msg: str) -> None:
        assert not parser.is_confirm(msg)

    @pytest.mark.parametrize(
        "msg",
        [
            "pode apagar",
            "Sim, pode apagar",
            "sí, borra",
            "yes, delete it",
            "puedes borrar",
            "puede eliminar",
        ],
    )
    def test_unnegated_restatements_still_confirm(
        self, parser: RegexConfirmParser, msg: str
    ) -> None:
        assert parser.is_confirm(msg)
