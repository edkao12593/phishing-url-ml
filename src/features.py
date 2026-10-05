"""Extract lexical and structural features from URL strings."""

from collections import Counter
import ipaddress
import math
import re
from urllib.parse import urlsplit

import pandas as pd
import tldextract

# Bundled PSL snapshot only; private suffixes separate unrelated hosted tenants.
PSL = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)
TOKENS = ("login", "verify", "secure", "account", "update", "bank")
MAX_URL_LENGTH = 16384


def parse_url(value: str):
    """Accept HTTP(S) and schemeless input; reject controls and invalid authority."""
    if not isinstance(value, str):
        raise ValueError("URL must be text")
    value = value.strip()
    if not value or len(value) > MAX_URL_LENGTH or re.search(r"\s", value):
        raise ValueError(
            "URL must be nonempty, without whitespace, and <=16384 characters"
        )
    if value.startswith("//"):
        value = "http:" + value
    elif not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        # Do not reinterpret javascript:, data:, or mailto: as hostnames.
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) and not re.match(
            r"^[^/:]+:\d+(?:/|$)", value
        ):
            raise ValueError("Only HTTP(S) URLs are supported")
        value = "http://" + value
    p = urlsplit(value)
    if p.scheme.lower() not in ("http", "https") or not p.hostname:
        raise ValueError("Expected an HTTP(S) URL with a hostname")
    host = p.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    if not host or re.search(r"[^a-z0-9.\-:]", host):
        raise ValueError("Invalid hostname")
    _ = p.port  # validates port number
    return value, p, host


def canonical_url(value: str) -> str:
    """Lowercase/IDNA host and scheme; preserve path/query/fragment case and bytes."""
    _, p, h = parse_url(value)
    authority = ("[" + h + "]") if ":" in h else h
    if p.port is not None:
        authority += ":" + str(p.port)
    if p.username is not None:
        authority = p.netloc.rsplit("@", 1)[0] + "@" + authority
    return p._replace(scheme=p.scheme.lower(), netloc=authority).geturl()


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def domain_group(url: str) -> str:
    _, _, host = parse_url(url)
    if is_ip(host):
        return host
    ex = PSL(host)
    return ex.top_domain_under_public_suffix or host


def entropy(text: str) -> float:
    if not text:
        return 0.0
    return -sum(
        (n / len(text)) * math.log2(n / len(text)) for n in Counter(text).values()
    )


def extract_features(url: str) -> dict:
    u, p, host = parse_url(url)
    ex = PSL(host) if not is_ip(host) else None
    parts = [s for s in p.path.split("/") if s]
    digits = sum(c.isascii() and c.isdigit() for c in u)
    letters = sum(c.isascii() and c.isalpha() for c in u)
    f = dict(
        url_length=len(u),
        hostname_length=len(host),
        path_length=len(p.path),
        query_length=len(p.query),
        fragment_length=len(p.fragment),
        digit_count=digits,
        digit_ratio=digits / len(u),
        letter_ratio=letters / len(u),
        dot_count=u.count("."),
        hyphen_count=u.count("-"),
        underscore_count=u.count("_"),
        slash_count=u.count("/"),
        at_count=u.count("@"),
        percent_count=u.count("%"),
        ampersand_count=u.count("&"),
        equals_count=u.count("="),
        question_count=u.count("?"),
        colon_count=u.count(":"),
        special_ratio=sum(not c.isalnum() for c in u) / len(u),
        subdomain_count=(len(ex.subdomain.split(".")) if ex and ex.subdomain else 0),
        https=int(p.scheme.lower() == "https"),
        ip_host=int(is_ip(host)),
        has_port=int(p.port is not None),
        has_userinfo=int(p.username is not None),
        hostname_digit_ratio=sum(c.isdigit() for c in host) / len(host),
        hostname_hyphens=host.count("-"),
        punycode=int("xn--" in host),
        url_entropy=entropy(u),
        hostname_entropy=entropy(host),
        path_depth=len(parts),
        max_path_component=max(map(len, parts), default=0),
        max_host_component=max(map(len, host.split(".")), default=0),
        query_parameter_count=len(p.query.split("&")) if p.query else 0,
        encoded_octet_count=len(re.findall(r"%[0-9a-fA-F]{2}", u)),
        double_slash_in_path=int("//" in p.path),
        long_component=int(any(len(s) > 40 for s in parts + host.split("."))),
    )
    lower = u.lower()
    f.update({f"token_{t}": int(t in lower) for t in TOKENS})
    f["suspicious_token_count"] = sum(lower.count(t) for t in TOKENS)
    return f


FEATURE_NAMES = list(extract_features("https://example.org/a?b=1"))


def feature_frame(urls) -> pd.DataFrame:
    return pd.DataFrame(
        [extract_features(u) for u in urls], columns=FEATURE_NAMES
    ).astype(float)
