"""Deterministic URL transformations for robustness experiments."""

from urllib.parse import urlunsplit
from .features import parse_url, canonical_url


QUERY_PADDING = "ref=homepage&lang=en&session=20261003"


def perturb(url: str, kind: str) -> str:
    _, p, _ = parse_url(url)
    if kind == "query_padding":
        q = p.query + ("&" if p.query else "") + QUERY_PADDING
        p = p._replace(query=q)
    elif kind == "fragment_padding":
        p = p._replace(fragment=p.fragment + "-homepage-documentation-support-2026")
    elif kind == "benign_path":
        p = p._replace(path=p.path.rstrip("/") + "/help-center2026/documentation")
    elif kind == "path_repeat":
        p = p._replace(path=p.path.rstrip("/") + "/" + (p.path.strip("/") or "home"))
    elif kind == "subdomain_prefix":
        # Subdomain prefixes apply only to DNS hostnames.
        from .features import is_ip

        if is_ip(p.hostname or ""):
            return canonical_url(url)
        authority = p.netloc
        user, sep, host = authority.rpartition("@")
        p = p._replace(
            netloc=(user + sep if sep else "")
            + "www2.help-center."
            + (host if sep else authority)
        )
    elif kind == "percent_encode_path":
        # Encode ASCII letters while retaining existing percent escapes.
        path, i, count = "", 0, 0
        while i < len(p.path):
            c = p.path[i]
            if c == "%" and i + 2 < len(p.path):
                path += p.path[i : i + 3]
                i += 3
                continue
            if c.isascii() and c.isalpha() and count < 8:
                path += f"%{ord(c):02X}"
                count += 1
            else:
                path += c
            i += 1
        p = p._replace(path=path)
    else:
        raise ValueError("Unknown perturbation: " + kind)
    return canonical_url(urlunsplit(p))


TRAIN_TRANSFORMS = ("query_padding", "benign_path")
TEST_TRANSFORMS = (
    *TRAIN_TRANSFORMS,
    "fragment_padding",
    "path_repeat",
    "subdomain_prefix",
    "percent_encode_path",
)
HELD_OUT_TRANSFORMS = tuple(t for t in TEST_TRANSFORMS if t not in TRAIN_TRANSFORMS)
