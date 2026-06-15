"""
Shared SSRF (server-side request forgery) URL validation.

A single source of truth for the blocked-network list and the
hostname-resolution check used by every outbound-request surface in the
library (web search/scrape tools, headless-browser tools, RAG document
loaders, and the serve webhook dispatcher).

Two entry points expose the same logic under the two contracts callers
need:

- :func:`validate_url` returns an error string (or ``None`` when safe), for
  tool functions that report errors as their return value.
- :func:`check_url` raises :class:`ValueError`, for internal callers that
  prefer exceptions.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import List, Optional, Union
from urllib.parse import urlparse

# Private / loopback / link-local networks that must never be reached from a
# user- or model-supplied URL.
_BLOCKED_NETWORKS: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def validate_url(url: str) -> Optional[str]:
    """Validate *url* against SSRF rules.

    Returns a human-readable error message if the URL uses a non-HTTP
    scheme, has no hostname, targets a loopback/internal name, or resolves
    to a private/reserved IP range. Returns ``None`` when the URL is safe to
    request.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return f"Error: URL scheme {parsed.scheme!r} is not allowed."

    hostname = parsed.hostname
    if not hostname:
        return "Error: URL has no hostname."

    lower_host = hostname.lower()
    if lower_host in ("localhost", "0.0.0.0"):  # nosec B104 — comparison, not a bind
        return f"Error: Requests to {hostname!r} are blocked (loopback/internal address)."

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return f"Error: Could not resolve hostname {hostname!r}: {e}"

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                return (
                    f"Error: URL resolves to private/reserved address {ip} "
                    f"(network {network}). Requests to internal networks are blocked."
                )

    return None


def check_url(url: str) -> None:
    """Raise :class:`ValueError` if *url* fails SSRF validation.

    Equivalent to :func:`validate_url` but raises instead of returning the
    message (with the ``"Error: "`` prefix stripped).
    """
    message = validate_url(url)
    if message is not None:
        raise ValueError(message.removeprefix("Error: "))


__all__ = ["validate_url", "check_url"]
