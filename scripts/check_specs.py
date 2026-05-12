#!/usr/bin/env python3
"""Validate the lewm-rs specification set."""

from __future__ import annotations

import argparse
import ast
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


RFC_REQUIRED = {
    "rfc",
    "title",
    "status",
    "version",
    "authors",
    "reviewers",
    "created",
    "updated",
    "supersedes",
    "superseded_by",
    "tracks_prd",
    "depends_on",
    "related",
}
ADR_REQUIRED = {
    "adr",
    "title",
    "status",
    "date",
    "authors",
    "tracks_rfc",
    "supersedes",
    "superseded_by",
    "pr",
}

RFC_STATUSES = {"Draft", "Proposed", "Accepted", "Implemented", "Superseded", "Retired"}
ADR_STATUSES = {"Proposed", "Accepted", "Implemented", "Superseded", "Retired"}

FR_NFR_RE = re.compile(r"\b(?:FR|NFR)-\d{3}\b")
RFC_NUM_RE = re.compile(r"^\d{4}")
TST_ID_RE = re.compile(r"\bTST-\d{4}-[A-Z0-9][A-Z0-9-]*-\d{3}\b")
TST_RANGE_RE = re.compile(r"\b(TST-\d{4}-[A-Z0-9][A-Z0-9-]*-)(\d{3})\.\.(\d{3})\b")
TST_WILDCARD_RE = re.compile(r"\bTST-\d{4}-\*\b")
LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:[-/][A-Z0-9]+)*\b")

ALLOWED_ACRONYMS = {
    "AI",
    "AC",
    "AD",
    "AMP",
    "AND",
    "API",
    "ASCII",
    "AWS",
    "CHW",
    "BLAKE",
    "BOM",
    "CF",
    "CI",
    "CD",
    "CHANGELOG",
    "CODEOWNERS",
    "COOLDOWN",
    "CPU",
    "CRITICAL",
    "CRUD",
    "CSV",
    "CVSS",
    "CVE",
    "CUDA",
    "DB",
    "DEBUG",
    "DINO-WM",
    "DOI",
    "DONE",
    "DR",
    "DRAFT",
    "DX",
    "EOF",
    "ENV",
    "ERROR",
    "EVAL",
    "FFI",
    "FPS",
    "FS",
    "FSF",
    "FULL",
    "GB",
    "GELU",
    "GEMM",
    "GHCR",
    "GH",
    "GPU",
    "HEAD",
    "HIGH",
    "HTML",
    "HWC",
    "HTTP",
    "HTTPS",
    "ID",
    "IDE",
    "INIT",
    "INFO",
    "IOPS",
    "INV-ID",
    "JIT",
    "JSON-RPC",
    "JSON",
    "JSONL",
    "LF",
    "LFS",
    "LGTM",
    "LICENSE",
    "LLM",
    "LR",
    "MAY",
    "MB",
    "MCP",
    "MFA",
    "MHA",
    "MIT",
    "ML",
    "MPPI",
    "MP",
    "MUST",
    "NVIDIA",
    "NVDEC",
    "NVTX",
    "NOT",
    "OIDC",
    "OOM",
    "OPL",
    "OR",
    "OPTIONAL",
    "OS",
    "OSI",
    "ORT",
    "PATH",
    "PDF",
    "PII",
    "PIL",
    "PR",
    "QKV",
    "RAM",
    "README",
    "RECOMMENDED",
    "REQUIRED",
    "REQUIRES",
    "REST",
    "RGB",
    "RL",
    "RPC",
    "RUN",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
    "RUSTSEC",
    "SBOM",
    "SDK",
    "SECURITY",
    "SHA",
    "SHALL",
    "SHORT",
    "SHOULD",
    "SIGINT",
    "SIGTERM",
    "SIMD",
    "SMOKE",
    "STEADY",
    "TB",
    "TBD",
    "TL",
    "TUI",
    "TTY",
    "UI",
    "UID",
    "UPLOAD",
    "URI",
    "URL",
    "USD",
    "UTF",
    "UTC",
    "UX",
    "VALUE",
    "VRAM",
    "WARMUP",
    "WARN",
    "WORKDIR",
    "XML",
    "XL",
    "YAML",
}


@dataclass(frozen=True)
class SpecFile:
    path: Path
    frontmatter: dict[str, object]


class SpecError(Exception):
    """Raised when a spec validation check fails."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def strip_inline_comment(value: str) -> str:
    in_quote = False
    quote_char = ""
    for index, char in enumerate(value):
        if char in {'"', "'"} and (index == 0 or value[index - 1] != "\\"):
            if in_quote and char == quote_char:
                in_quote = False
                quote_char = ""
            elif not in_quote:
                in_quote = True
                quote_char = char
        if char == "#" and not in_quote:
            return value[:index].strip()
    return value.strip()


def parse_scalar(value: str) -> object:
    value = strip_inline_comment(value)
    if value == "null":
        return None
    if value in {"[]", "{}"}:
        return ast.literal_eval(value)
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def parse_frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("<!--"):
        comment_end = stripped.find("-->")
        if comment_end == -1:
            raise SpecError(f"{path}: unterminated leading HTML comment")
        text = stripped[comment_end + 3 :].lstrip()
    else:
        text = stripped
    if not text.startswith("---\n"):
        raise SpecError(f"{path}: missing YAML frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise SpecError(f"{path}: unterminated YAML frontmatter")

    data: dict[str, object] = {}
    for line_no, line in enumerate(text[4:end].splitlines(), start=2):
        if not line.strip():
            continue
        if ":" not in line:
            raise SpecError(f"{path}:{line_no}: invalid frontmatter line")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise SpecError(f"{path}:{line_no}: empty frontmatter key")
        data[key] = parse_scalar(raw_value.strip())
    return data


def rfc_files(root: Path) -> list[Path]:
    return sorted(root.glob("specs/rfcs/[0-9][0-9][0-9][0-9]-*.md"))


def adr_files(root: Path) -> list[Path]:
    return sorted(root.glob("specs/adr/[0-9][0-9][0-9][0-9]-*.md"))


def markdown_files(root: Path) -> list[Path]:
    return [root / "PRD.md", *sorted((root / "specs").rglob("*.md"))]


def rfc_number(path: Path) -> str:
    match = RFC_NUM_RE.match(path.name)
    if not match:
        raise SpecError(f"{path}: RFC filename must start with a four-digit number")
    return match.group(0)


def load_rfc_specs(root: Path) -> dict[str, SpecFile]:
    specs = {}
    for path in rfc_files(root):
        number = rfc_number(path)
        if number in specs:
            raise SpecError(f"duplicate RFC number {number}")
        specs[number] = SpecFile(path, parse_frontmatter(path))
    return specs


def check_frontmatter(root: Path) -> None:
    for number, spec in load_rfc_specs(root).items():
        missing = RFC_REQUIRED - set(spec.frontmatter)
        if missing:
            raise SpecError(f"{spec.path}: missing RFC frontmatter keys: {sorted(missing)}")
        if spec.frontmatter["rfc"] != number:
            raise SpecError(f"{spec.path}: frontmatter rfc does not match filename")
        if spec.frontmatter["status"] not in RFC_STATUSES:
            raise SpecError(f"{spec.path}: invalid RFC status {spec.frontmatter['status']!r}")
        for key in ("authors", "reviewers", "supersedes", "tracks_prd", "depends_on", "related"):
            if not isinstance(spec.frontmatter[key], list):
                raise SpecError(f"{spec.path}: frontmatter {key!r} must be a list")

    for path in adr_files(root):
        frontmatter = parse_frontmatter(path)
        missing = ADR_REQUIRED - set(frontmatter)
        if missing:
            raise SpecError(f"{path}: missing ADR frontmatter keys: {sorted(missing)}")
        number = rfc_number(path)
        if frontmatter["adr"] != number:
            raise SpecError(f"{path}: frontmatter adr does not match filename")
        if frontmatter["status"] not in ADR_STATUSES:
            raise SpecError(f"{path}: invalid ADR status {frontmatter['status']!r}")


def slugify_heading(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    allowed = set(string.ascii_lowercase + string.digits + " -_")
    text = "".join(char for char in text if char in allowed)
    return re.sub(r"-+", "-", re.sub(r"\s+", "-", text)).strip("-")


def headings(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    anchors = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        slug = slugify_heading(match.group(2))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def without_fenced_code(text: str) -> str:
    kept = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append(line)
    return "\n".join(kept)


def check_links(root: Path) -> None:
    heading_cache: dict[Path, set[str]] = {}
    for path in markdown_files(root):
        text = without_fenced_code(path.read_text(encoding="utf-8"))
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().split()[0]
            if "{{" in target or "}}" in target:
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                target_path = path
                anchor = target[1:]
            else:
                target_without_anchor, _, anchor = target.partition("#")
                if not target_without_anchor:
                    target_path = path
                else:
                    target_path = (path.parent / unquote(target_without_anchor)).resolve()
            if not target_path.exists():
                raise SpecError(f"{path}: local link target does not exist: {raw_target}")
            if anchor and target_path.suffix == ".md":
                anchors = heading_cache.setdefault(target_path, headings(target_path))
                if anchor not in anchors:
                    rel = target_path.relative_to(root)
                    raise SpecError(f"{path}: anchor #{anchor} not found in {rel}")


def expand_test_ranges(text: str) -> set[str]:
    ids = set(TST_ID_RE.findall(text))
    for prefix, start, end in TST_RANGE_RE.findall(text):
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            raise SpecError(f"invalid test ID range: {prefix}{start}..{end}")
        for value in range(start_i, end_i + 1):
            ids.add(f"{prefix}{value:03d}")
    return ids


def check_traceability(root: Path) -> None:
    prd_ids = set(FR_NFR_RE.findall((root / "PRD.md").read_text(encoding="utf-8")))
    matrix_text = (root / "specs/traceability-matrix.md").read_text(encoding="utf-8")
    matrix_ids = set(FR_NFR_RE.findall(matrix_text))
    missing = sorted(prd_ids - matrix_ids)
    if missing:
        raise SpecError(f"traceability matrix missing PRD requirement IDs: {missing}")

    rfc_tests: set[str] = set()
    for path in rfc_files(root):
        if path.name.startswith("0000-"):
            continue
        rfc_tests |= expand_test_ranges(path.read_text(encoding="utf-8"))

    matrix_tests = expand_test_ranges(matrix_text)
    missing_tests = sorted(test_id for test_id in matrix_tests if test_id not in rfc_tests)
    if missing_tests:
        raise SpecError(f"traceability matrix references missing RFC test IDs: {missing_tests}")

    for wildcard in TST_WILDCARD_RE.findall(matrix_text):
        prefix = wildcard[:-1]
        if not any(test_id.startswith(prefix) for test_id in rfc_tests):
            raise SpecError(f"traceability matrix wildcard has no matching RFC tests: {wildcard}")


def glossary_terms(root: Path) -> set[str]:
    text = (root / "specs/glossary.md").read_text(encoding="utf-8")
    terms = set(re.findall(r"^\*\*([^*]+)\*\*\s+—", text, flags=re.MULTILINE))
    terms.update(re.findall(r"^- \*\*([^*]+)\*\*", text, flags=re.MULTILINE))
    return terms


def check_glossary(root: Path) -> None:
    terms = glossary_terms(root)
    acronyms = {term for term in terms if term.upper() == term}
    allowed = acronyms | ALLOWED_ACRONYMS
    failures: dict[str, set[str]] = {}
    for path in markdown_files(root):
        if path.name == "glossary.md":
            continue
        text = without_fenced_code(path.read_text(encoding="utf-8"))
        text = re.sub(r"`[^`]+`", "", text)
        found = {token for token in ACRONYM_RE.findall(text) if len(token) > 1}
        unknown = sorted(
            token
            for token in found
            if token not in allowed
            and not any(char.isdigit() for char in token)
            and "/" not in token
            and not token.endswith("-NNN")
        )
        if unknown:
            failures[str(path.relative_to(root))] = set(unknown)

    if failures:
        details = "; ".join(f"{path}: {sorted(tokens)}" for path, tokens in sorted(failures.items()))
        raise SpecError(f"glossary missing acronym definitions or allow-list entries: {details}")


def check_rfc_numbering(root: Path) -> None:
    numbers = sorted(int(number) for number in load_rfc_specs(root))
    if not numbers or numbers[0] != 0:
        raise SpecError("RFC numbering must include 0000 template")
    expected = list(range(0, numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise SpecError(f"RFC numbering has gaps: {missing}")


def check_status_lifecycle(root: Path) -> None:
    specs = load_rfc_specs(root)
    accepted = {
        number
        for number, spec in specs.items()
        if spec.frontmatter.get("status") in {"Accepted", "Implemented"}
    }
    for number, spec in specs.items():
        if number == "0000" or spec.frontmatter.get("status") not in {"Accepted", "Implemented"}:
            continue
        refs: set[str] = set()
        for key in ("depends_on", "related", "supersedes"):
            values = spec.frontmatter.get(key, [])
            if isinstance(values, list):
                refs.update(str(value) for value in values)
        refs.discard("0000")
        missing = sorted(ref for ref in refs if ref not in specs)
        if missing:
            raise SpecError(f"{spec.path}: references unknown RFCs: {missing}")
        non_accepted = sorted(ref for ref in refs if ref not in accepted)
        if non_accepted:
            raise SpecError(f"{spec.path}: accepted RFC references non-accepted RFCs: {non_accepted}")


def run_checks(root: Path, checks: list[str]) -> None:
    check_map = {
        "frontmatter": check_frontmatter,
        "links": check_links,
        "traceability": check_traceability,
        "glossary": check_glossary,
        "rfc-numbering": check_rfc_numbering,
        "status-lifecycle": check_status_lifecycle,
    }
    for check in checks:
        check_map[check](root)
        print(f"check_specs: {check} ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-frontmatter", action="store_true")
    parser.add_argument("--check-links", action="store_true")
    parser.add_argument("--check-traceability", action="store_true")
    parser.add_argument("--check-glossary", action="store_true")
    parser.add_argument("--check-rfc-numbering", action="store_true")
    parser.add_argument("--check-status-lifecycle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = []
    if args.check_frontmatter:
        selected.append("frontmatter")
    if args.check_links:
        selected.append("links")
    if args.check_traceability:
        selected.append("traceability")
    if args.check_glossary:
        selected.append("glossary")
    if args.check_rfc_numbering:
        selected.append("rfc-numbering")
    if args.check_status_lifecycle:
        selected.append("status-lifecycle")
    if not selected:
        selected = [
            "frontmatter",
            "links",
            "traceability",
            "glossary",
            "rfc-numbering",
            "status-lifecycle",
        ]

    try:
        run_checks(repo_root(), selected)
    except (OSError, SpecError, ValueError, SyntaxError) as error:
        print(f"check_specs: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
