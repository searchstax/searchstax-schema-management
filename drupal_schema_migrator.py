#!/usr/bin/env python3
"""
Drupal → SearchStax Site Search Schema Migrator (MVP)

Goals (MVP):
- Read local Drupal Search API Solr XMLs (schema.xml, schema_extra_types.xml, schema_extra_fields.xml)
- Fetch current target schema from SearchStax (Schema API)
- Generate a plan (additive-only): create missing fieldTypes/fields/dynamicFields/copyFields
- Surgical fieldType update: if a fieldType exists on both sides and only *additional* analyzer factories from Drupal are missing on target,
  stage a replace of the target fieldType with those factories inserted at the *same indices* as in Drupal. We never remove/reorder existing target pieces.
- Apply: execute creates and the safe replace-field-type operations (optional --yes gate)

Notes:
- This is intentionally an MVP—light on error handling; extend as needed.
- Requires: typer, requests, pyyaml, lxml

Usage examples:
  python drupal_schema_migrator.py audit --drupal-config ./conf --target-url https://HOST/solr/COLLECTION --api-key XXX
  python drupal_schema_migrator.py plan  --drupal-config ./conf --target-url https://HOST/solr/COLLECTION --api-key XXX --out plan.yaml
  python drupal_schema_migrator.py apply --plan plan.yaml --target-url https://HOST/solr/COLLECTION --api-key XXX --yes
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Tuple

import typer
import requests
import yaml
from lxml import etree as ET

import difflib
import html as html_lib


app = typer.Typer(
    add_completion=False, help="Drupal → SearchStax Schema Migrator (MVP)"
)

# -----------------------------
# Models
# -----------------------------


@dataclass
class Factory:
    clazz: str
    params: Dict[str, Any] = dataclass_field(default_factory=dict)

    def identity(self) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        # identity used for equality and presence checks
        c = (self.clazz or "").strip().lower()
        items = tuple(sorted((str(k).lower(), str(v)) for k, v in self.params.items()))
        return (c, items)

    def to_json(self) -> Dict[str, Any]:
        d = {"class": self.clazz}
        d.update(self.params)
        return d


@dataclass
class Analyzer:
    charfilters: List[Factory] = dataclass_field(default_factory=list)
    tokenizer: Optional[Factory] = None
    filters: List[Factory] = dataclass_field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.charfilters:
            out["charFilters"] = [f.to_json() for f in self.charfilters]
        if self.tokenizer:
            out["tokenizer"] = self.tokenizer.to_json()
        if self.filters:
            out["filters"] = [f.to_json() for f in self.filters]
        return out


@dataclass
class FieldType:
    name: str
    clazz: str
    analyzer_index: Optional[Analyzer] = None
    analyzer_query: Optional[Analyzer] = None
    attrs: Dict[str, Any] = dataclass_field(default_factory=dict)

    def to_replace_payload(self) -> Dict[str, Any]:
        body = {"name": self.name, "class": self.clazz}
        if self.analyzer_index:
            body["indexAnalyzer"] = self.analyzer_index.to_json()
        if self.analyzer_query:
            body["queryAnalyzer"] = self.analyzer_query.to_json()
        for k in ("positionIncrementGap", "omitNorms"):
            if k in self.attrs:
                body[k] = self.attrs[k]
        return {"replace-field-type": body}


@dataclass
class Field:
    name: str
    type: str
    indexed: bool = True
    stored: bool = True
    multiValued: bool = False
    docValues: Optional[bool] = None
    attrs: Dict[str, Any] = dataclass_field(default_factory=dict)

    def to_add_payload(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.type,
            "indexed": self.indexed,
            "stored": self.stored,
        }
        if self.multiValued is not None:
            d["multiValued"] = self.multiValued
        if self.docValues is not None:
            d["docValues"] = self.docValues
        # pass through a few safe attrs if present
        for k in ("omitNorms", "termVectors"):
            if k in self.attrs:
                d[k] = self.attrs[k]
        return {"add-field": d}


@dataclass
class DynamicField(Field):
    def to_add_payload(self) -> Dict[str, Any]:
        d = super().to_add_payload()["add-field"]
        return {"add-dynamic-field": d}


@dataclass
class CopyField:
    source: str
    dest: str
    maxChars: Optional[int] = None

    def to_add_payload(self) -> Dict[str, Any]:
        d = {"source": self.source, "dest": self.dest}
        if self.maxChars is not None:
            d["maxChars"] = self.maxChars
        return {"add-copy-field": d}


@dataclass
class SchemaModel:
    uniqueKey: Optional[str] = None
    fieldTypes: Dict[str, FieldType] = dataclass_field(default_factory=dict)
    fields: Dict[str, Field] = dataclass_field(default_factory=dict)
    dynamicFields: Dict[str, DynamicField] = dataclass_field(
        default_factory=dict
    )  # keyed by name pattern
    copyFields: List[CopyField] = dataclass_field(default_factory=list)


# -----------------------------
# XML Parsing (Drupal configs)
# -----------------------------


def _bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).lower() in ("1", "true", "yes")


def parse_factory(el: ET._Element) -> Factory:
    clazz = el.attrib.get("class", "").strip()
    params = {k: v for k, v in el.attrib.items() if k != "class"}
    return Factory(clazz=clazz, params=params)


def parse_analyzer(ft_el: ET._Element) -> Analyzer:
    # fieldType may have <analyzer>, or separate <analyzer type="index"> and type="query" (we'll treat single analyzer only in MVP)
    analyzer_el = ft_el.find("analyzer")
    if analyzer_el is None:
        # try index analyzer first
        analyzer_el = ft_el.xpath("analyzer[@type='index']")
        if analyzer_el:
            analyzer_el = analyzer_el[0]
        else:
            analyzer_el = ft_el.xpath("analyzer[@type='query']")
            analyzer_el = analyzer_el[0] if analyzer_el else None
    if analyzer_el is None:
        return Analyzer()

    # charFilters, tokenizer, filters in document order
    cf = [parse_factory(x) for x in analyzer_el.findall("charFilter")]
    tok_el = analyzer_el.find("tokenizer")
    tok = parse_factory(tok_el) if tok_el is not None else None
    filts = [parse_factory(x) for x in analyzer_el.findall("filter")]
    return Analyzer(charfilters=cf, tokenizer=tok, filters=filts)


def parse_fieldtype(ft_el: ET._Element) -> FieldType:
    name = ft_el.attrib["name"]
    clazz = ft_el.attrib.get("class", "")
    attrs = {k: v for k, v in ft_el.attrib.items() if k not in ("name", "class")}
    idx_an, qry_an = parse_analyzers(ft_el)
    return FieldType(
        name=name,
        clazz=clazz,
        analyzer_index=idx_an,
        analyzer_query=qry_an,
        attrs=attrs,
    )


def parse_field_like(el: ET._Element) -> Field:
    a = el.attrib
    return Field(
        name=a["name"],
        type=a["type"],
        indexed=_bool(a.get("indexed"), True),
        stored=_bool(a.get("stored"), True),
        multiValued=_bool(a.get("multiValued"), False),
        docValues=(
            _bool(a.get("docValues"), None) if a.get("docValues") is not None else None
        ),
        attrs={
            k: v
            for k, v in a.items()
            if k
            not in ("name", "type", "indexed", "stored", "multiValued", "docValues")
        },
    )


def parse_copyfield(el: ET._Element) -> CopyField:
    a = el.attrib
    mc = int(a["maxChars"]) if a.get("maxChars") is not None else None
    return CopyField(source=a["source"], dest=a["dest"], maxChars=mc)


def _parse_single_analyzer(an_el) -> Analyzer:
    if an_el is None:
        return Analyzer()
    cf = [parse_factory(x) for x in an_el.findall("charFilter")]
    tok_el = an_el.find("tokenizer")
    tok = parse_factory(tok_el) if tok_el is not None else None
    filts = [parse_factory(x) for x in an_el.findall("filter")]
    return Analyzer(charfilters=cf, tokenizer=tok, filters=filts)


def parse_analyzers(ft_el: ET._Element) -> tuple[Analyzer, Analyzer]:
    # Prefer explicit index/query analyzers; fall back to a single <analyzer>
    idx = ft_el.xpath("analyzer[@type='index']")
    qry = ft_el.xpath("analyzer[@type='query']")
    if idx or qry:
        idx_an = _parse_single_analyzer(idx[0] if idx else None)
        qry_an = _parse_single_analyzer(qry[0] if qry else None)
        return idx_an, qry_an
    # single analyzer applies to both
    single = ft_el.find("analyzer")
    an = _parse_single_analyzer(single)
    return an, an


def _detect_required_entities(schema_path: str) -> list[str]:
    """
    Scan schema.xml for external entity declarations and return a list of files
    it expects to include, e.g., ["schema_extra_fields.xml", "schema_extra_types.xml"].
    Best-effort regex against the DOCTYPE internal subset.
    """
    required: list[str] = []
    try:
        with open(schema_path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except FileNotFoundError:
        return required

    import re

    # Matches: <!ENTITY extrafields SYSTEM "schema_extra_fields.xml">
    pattern = re.compile(r'<!ENTITY\s+[^>]*?SYSTEM\s+"([^"]+)"\s*>', re.IGNORECASE)
    for m in pattern.finditer(text):
        candidate = m.group(1).strip()
        if candidate.lower().endswith(".xml"):
            required.append(candidate)
    return required


def load_drupal_schema(dirpath: str) -> SchemaModel:
    # Enable external entity resolution so schema.xml can include &extrafields; &extratypes; etc.
    parser = ET.XMLParser(
        load_dtd=True,
        resolve_entities=True,
        no_network=True,
        recover=True,
        huge_tree=True,
    )

    # Pre-flight: if schema.xml declares external entities, verify the referenced files exist.
    schema_main = os.path.join(dirpath, "schema.xml")
    missing: list[str] = []
    for rel in _detect_required_entities(schema_main):
        candidate = os.path.join(dirpath, rel)
        if not os.path.exists(candidate):
            missing.append(f"{rel}  (expected at: {candidate})")
    if missing:
        raise FileNotFoundError(
            "schema.xml references external entities that were not found:\n  - "
            + "\n  - ".join(missing)
            + "\nPlease ensure these files are present alongside schema.xml, or update the paths in the DOCTYPE."
        )

    paths = [
        os.path.join(dirpath, "schema.xml"),
        os.path.join(dirpath, "schema_extra_types.xml"),
        os.path.join(dirpath, "schema_extra_fields.xml"),
    ]
    model = SchemaModel()
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            tree = ET.parse(p, parser=parser)
        except ET.XMLSyntaxError as e:
            raise ET.XMLSyntaxError(
                f"Failed to parse '{p}'. If you are using &extrafields; or &extratypes;, "
                f"ensure the referenced files exist in '{dirpath}'. Original error: {e}",
                e.error_log,
            )
        root = tree.getroot()

        uk = root.find("uniqueKey")
        if uk is not None and uk.text:
            model.uniqueKey = (uk.text or "").strip()

        for ft in root.findall(".//fieldType"):
            fto = parse_fieldtype(ft)
            model.fieldTypes[fto.name] = fto

        for f in root.findall(".//field"):
            fld = parse_field_like(f)
            model.fields[fld.name] = fld

        for df in root.findall(".//dynamicField"):
            dfo = parse_field_like(df)
            model.dynamicFields[dfo.name] = DynamicField(**dfo.__dict__)

        for cf in root.findall(".//copyField"):
            model.copyFields.append(parse_copyfield(cf))

    return model


# -----------------------------
# Target (Site Search) Schema API client (very small)
# -----------------------------


def _normalize_target_url(url: str) -> str:
    """Accepts either a base collection URL or a full /update (or /schema) endpoint and
    normalizes it to the collection base (without trailing /schema or /update).
    Examples:
      https://host/solr/collection            -> https://host/solr/collection
      https://host/solr/collection/update     -> https://host/solr/collection
      https://host/solr/collection/schema     -> https://host/solr/collection
    """
    u = url.rstrip("/")
    for suffix in ("/update", "/schema"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    return u


class SchemaAPI:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        # Accepts base collection URL or full /update or /schema URL and normalizes it
        self.base_url = _normalize_target_url(base_url)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            # SearchStax uses Token-based auth: Authorization: Token <API_KEY>
            self.session.headers.update({"Authorization": f"Token {api_key}"})
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}/schema{path}"

    def get_full_schema(self) -> Dict[str, Any]:
        # Use dedicated endpoints to reduce payload size if needed; MVP uses /schema
        r = self.session.get(self._url(""), timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("schema", r.json())

    def post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            self._url(""), data=json.dumps(payload), timeout=self.timeout
        )
        try:
            r.raise_for_status()
        except Exception as e:
            typer.echo(f"Schema API error: {r.status_code} {r.text}")
            raise
        return r.json()


# -----------------------------
# Normalization & Diff helpers
# -----------------------------


def _analyzer_to_json(an: Analyzer) -> dict:
    return (an or Analyzer()).to_json()


def _pretty_lines(obj: dict) -> list[str]:
    return json.dumps(obj, indent=2, sort_keys=True).splitlines()


def _field_to_json(f: Field) -> dict:
    if not f:
        return {}
    base = {
        "name": f.name,
        "type": f.type,
        "indexed": f.indexed,
        "stored": f.stored,
    }
    if f.multiValued is not None:
        base["multiValued"] = f.multiValued
    if f.docValues is not None:
        base["docValues"] = f.docValues
    if f.attrs:
        base["attrs"] = dict(sorted(f.attrs.items()))
    return base


def _dynamic_to_json(df: DynamicField) -> dict:
    # Same shape as fields, just keep the name pattern
    return _field_to_json(df)


def _h(title: str) -> str:
    return f"<h2 style='margin:24px 0 8px'>{html_lib.escape(title)}</h2>"


def _p(text: str) -> str:
    return f"<p>{html_lib.escape(text)}</p>"


def analyzer_presence(dr: Analyzer, tg: Analyzer) -> Dict[str, Any]:
    """Return what pieces from Drupal analyzer are missing in target, preserving Drupal indices.
    We only consider *inserts*—we do not propose deletions or reorders.
    """
    inserts = {"charFilters": [], "filters": []}
    # CharFilters
    tg_ids = [f.identity() for f in tg.charfilters]
    for idx, f in enumerate(dr.charfilters):
        if f.identity() not in tg_ids:
            inserts["charFilters"].append((idx, f))
    # Tokenizer must match exactly; else we abort update path
    tok_mismatch = False
    if (dr.tokenizer and not tg.tokenizer) or (tg.tokenizer and not dr.tokenizer):
        tok_mismatch = True
    elif dr.tokenizer and tg.tokenizer:
        tok_mismatch = dr.tokenizer.identity() != tg.tokenizer.identity()
    # Filters
    tg_f_ids = [f.identity() for f in tg.filters]
    for idx, f in enumerate(dr.filters):
        if f.identity() not in tg_f_ids:
            inserts["filters"].append((idx, f))
    return {"inserts": inserts, "tokenizer_mismatch": tok_mismatch}


def build_replace_analyzer(dr: Analyzer, tg: Analyzer) -> Optional[Analyzer]:
    diff = analyzer_presence(dr, tg)
    if diff["tokenizer_mismatch"]:
        return None
    new_cf = list(tg.charfilters)
    for idx, fac in diff["inserts"]["charFilters"]:
        ins_at = min(idx, len(new_cf))
        new_cf.insert(ins_at, fac)
    new_filters = list(tg.filters)
    for idx, fac in diff["inserts"]["filters"]:
        ins_at = min(idx, len(new_filters))
        new_filters.insert(ins_at, fac)
    return Analyzer(charfilters=new_cf, tokenizer=tg.tokenizer, filters=new_filters)


def _analyzers_equal(a: Analyzer, b: Analyzer) -> bool:
    if (a.tokenizer is None) != (b.tokenizer is None):
        return False
    if a.tokenizer and b.tokenizer and a.tokenizer.identity() != b.tokenizer.identity():
        return False
    if [f.identity() for f in a.charfilters] != [f.identity() for f in b.charfilters]:
        return False
    if [f.identity() for f in a.filters] != [f.identity() for f in b.filters]:
        return False
    return True


def models_equal_fieldtype(ft_a: FieldType, ft_b: FieldType) -> bool:
    if (ft_a.clazz or "").lower() != (ft_b.clazz or "").lower():
        return False
    return _analyzers_equal(
        ft_a.analyzer_index or Analyzer(), ft_b.analyzer_index or Analyzer()
    ) and _analyzers_equal(
        ft_a.analyzer_query or Analyzer(), ft_b.analyzer_query or Analyzer()
    )


# -----------------------------
# Converters from Target JSON → SchemaModel (subset for what we need)
# -----------------------------


def target_to_model(target_schema: Dict[str, Any]) -> SchemaModel:
    m = SchemaModel(uniqueKey=target_schema.get("uniqueKey"))
    # fieldTypes
    for ft in target_schema.get("fieldTypes", []):
        name = ft.get("name")
        clazz = ft.get("class", "")
        attrs = {
            k: v
            for k, v in ft.items()
            if k not in ("name", "class", "analyzer", "indexAnalyzer", "queryAnalyzer")
        }

        # Prefer explicit index/queryAnalyzer; fall back to flat analyzer if present
        def _mk_an(an):
            if not an:
                return Analyzer()
            cf = [
                Factory(
                    x.get("class", ""), {k: v for k, v in x.items() if k != "class"}
                )
                for x in an.get("charFilters", [])
            ]
            tok = an.get("tokenizer")
            tokf = (
                Factory(
                    tok.get("class", ""), {k: v for k, v in tok.items() if k != "class"}
                )
                if tok
                else None
            )
            fil = [
                Factory(
                    x.get("class", ""), {k: v for k, v in x.items() if k != "class"}
                )
                for x in an.get("filters", [])
            ]
            return Analyzer(cf, tokf, fil)

        idx_an = _mk_an(ft.get("indexAnalyzer") or ft.get("analyzer"))
        qry_an = _mk_an(ft.get("queryAnalyzer") or ft.get("analyzer"))
        m.fieldTypes[name] = FieldType(
            name=name,
            clazz=clazz,
            analyzer_index=idx_an,
            analyzer_query=qry_an,
            attrs=attrs,
        )

    # fields
    for f in target_schema.get("fields", []):
        m.fields[f["name"]] = Field(
            name=f["name"],
            type=f.get("type", ""),
            indexed=f.get("indexed", True),
            stored=f.get("stored", True),
            multiValued=f.get("multiValued", False),
            docValues=f.get("docValues"),
            attrs={
                k: v
                for k, v in f.items()
                if k
                not in ("name", "type", "indexed", "stored", "multiValued", "docValues")
            },
        )
    # dynamicFields
    for df in target_schema.get("dynamicFields", []):
        base = Field(
            name=df["name"],
            type=df.get("type", ""),
            indexed=df.get("indexed", True),
            stored=df.get("stored", True),
            multiValued=df.get("multiValued", False),
            docValues=df.get("docValues"),
            attrs={
                k: v
                for k, v in df.items()
                if k
                not in ("name", "type", "indexed", "stored", "multiValued", "docValues")
            },
        )
        m.dynamicFields[base.name] = DynamicField(**base.__dict__)
    # copyFields
    for cf in target_schema.get("copyFields", []):
        m.copyFields.append(
            CopyField(
                source=cf.get("source"),
                dest=cf.get("dest"),
                maxChars=cf.get("maxChars"),
            )
        )
    return m


# -----------------------------
# Plan building (additive-only + surgical fieldType replace)
# -----------------------------


def build_plan(drupal: SchemaModel, target: SchemaModel) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "mode": "additive",
        "create_fieldTypes": [],
        "replace_fieldTypes": [],
        "create_fields": [],
        "create_dynamicFields": [],
        "add_copyFields": [],
        "notes": [],
    }

    # FieldTypes: exact equal → skip; else if name exists → try surgical replace; else create
    for name, d_ft in drupal.fieldTypes.items():
        t_ft = target.fieldTypes.get(name)
        if not t_ft:
            # (create logic unchanged)
            ...
        elif not models_equal_fieldtype(d_ft, t_ft):
            idx_merged = build_replace_analyzer(
                d_ft.analyzer_index or Analyzer(), t_ft.analyzer_index or Analyzer()
            )
            qry_merged = build_replace_analyzer(
                d_ft.analyzer_query or Analyzer(), t_ft.analyzer_query or Analyzer()
            )
            if idx_merged is None or qry_merged is None:
                plan["notes"].append(
                    f"Tokenizer mismatch for fieldType '{name}'; skipping replace. You may create a shadow type manually if desired."
                )
                continue
            new_ft = FieldType(
                name=name,
                clazz=t_ft.clazz or d_ft.clazz,
                analyzer_index=idx_merged,
                analyzer_query=qry_merged,
                attrs=t_ft.attrs or {},
            )
            plan["replace_fieldTypes"].append(new_ft.to_replace_payload())

    # Fields
    for name, f in drupal.fields.items():
        if name not in target.fields:
            plan["create_fields"].append(f.to_add_payload())

    # DynamicFields
    for name, df in drupal.dynamicFields.items():
        if name not in target.dynamicFields:
            plan["create_dynamicFields"].append(df.to_add_payload())

    # CopyFields (existence check by tuple)
    tgt_cf_set = set((c.source, c.dest, c.maxChars) for c in target.copyFields)
    for cf in drupal.copyFields:
        key = (cf.source, cf.dest, cf.maxChars)
        if key not in tgt_cf_set:
            plan["add_copyFields"].append(cf.to_add_payload())

    return plan


def generate_html_report(
    dr: SchemaModel, tg: SchemaModel, plan: dict, out_path: str
) -> None:
    hd = difflib.HtmlDiff(wrapcolumn=100)
    parts: list[str] = []
    parts.append(
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:ui-sans-serif,system-ui,Segoe UI,Arial}"
        "table{font-size:12px} .diff_add{background:#e6ffed} .diff_chg{background:#fff5b1} .diff_sub{background:#ffeef0}"
        "h2{margin:24px 0 8px} h3{margin:12px 0 6px}</style>"
        "<title>Schema Diff Report</title></head><body>"
    )
    parts.append("<h1>Schema Diff Report</h1>")
    parts.append(
        _p(
            f"create_fieldTypes: {len(plan.get('create_fieldTypes', []))} • "
            f"replace_fieldTypes: {len(plan.get('replace_fieldTypes', []))} • "
            f"create_fields: {len(plan.get('create_fields', []))} • "
            f"create_dynamicFields: {len(plan.get('create_dynamicFields', []))} • "
            f"add_copyFields: {len(plan.get('add_copyFields', []))}"
        )
    )
    if plan.get("notes"):
        parts.append(_h("Notes"))
        parts.append(
            "<ul>"
            + "".join(f"<li>{html_lib.escape(n)}</li>" for n in plan["notes"])
            + "</ul>"
        )

    # ------------------------
    # FieldType analyzer diffs
    # ------------------------
    common_ft = sorted(set(dr.fieldTypes.keys()) & set(tg.fieldTypes.keys()))
    for name in common_ft:
        d_ft = dr.fieldTypes[name]
        t_ft = tg.fieldTypes[name]

        sections = [
            (
                "indexAnalyzer",
                d_ft.analyzer_index or Analyzer(),
                t_ft.analyzer_index or Analyzer(),
            ),
            (
                "queryAnalyzer",
                d_ft.analyzer_query or Analyzer(),
                t_ft.analyzer_query or Analyzer(),
            ),
        ]
        show_any = False
        tables_html = []
        for label, d_an, t_an in sections:
            d_lines = _pretty_lines(_analyzer_to_json(d_an))
            t_lines = _pretty_lines(_analyzer_to_json(t_an))
            table = hd.make_table(
                d_lines,
                t_lines,
                fromdesc=f"Drupal {label}",
                todesc=f"Target {label}",
                context=True,
                numlines=3,
            )
            if d_lines != t_lines:
                show_any = True
            tables_html.append(f"<h3>{label}</h3>{table}")

        if show_any:
            parts.append(_h(f"fieldType: {name}"))
            parts.append(
                _p(f"class: Drupal={d_ft.clazz or ''} | Target={t_ft.clazz or ''}")
            )
            parts.extend(tables_html)

    # ------------------------
    # Fields diffs
    # ------------------------
    parts.append(_h("Fields"))
    dr_names = set(dr.fields.keys())
    tg_names = set(tg.fields.keys())
    only_in_dr = sorted(dr_names - tg_names)
    only_in_tg = sorted(tg_names - dr_names)
    if only_in_dr:
        parts.append(
            "<h3>Missing on Target (will be created)</h3><ul>"
            + "".join(f"<li>{html_lib.escape(n)}</li>" for n in only_in_dr)
            + "</ul>"
        )
    if only_in_tg:
        parts.append(
            "<h3>Only on Target (left unchanged)</h3><ul>"
            + "".join(f"<li>{html_lib.escape(n)}</li>" for n in only_in_tg)
            + "</ul>"
        )

    common_fields = sorted(dr_names & tg_names)
    for name in common_fields:
        d_f = dr.fields[name]
        t_f = tg.fields[name]
        d_lines = _pretty_lines(_field_to_json(d_f))
        t_lines = _pretty_lines(_field_to_json(t_f))
        if d_lines != t_lines:
            parts.append(f"<h3>{html_lib.escape(name)}</h3>")
            parts.append(
                hd.make_table(
                    d_lines,
                    t_lines,
                    fromdesc="Drupal field",
                    todesc="Target field",
                    context=True,
                    numlines=2,
                )
            )

    # ------------------------
    # DynamicFields diffs
    # ------------------------
    parts.append(_h("Dynamic Fields"))
    dr_dyn = set(dr.dynamicFields.keys())
    tg_dyn = set(tg.dynamicFields.keys())
    only_dyn_dr = sorted(dr_dyn - tg_dyn)
    only_dyn_tg = sorted(tg_dyn - dr_dyn)
    if only_dyn_dr:
        parts.append(
            "<h3>Missing on Target (will be created)</h3><ul>"
            + "".join(f"<li>{html_lib.escape(n)}</li>" for n in only_dyn_dr)
            + "</ul>"
        )
    if only_dyn_tg:
        parts.append(
            "<h3>Only on Target (left unchanged)</h3><ul>"
            + "".join(f"<li>{html_lib.escape(n)}</li>" for n in only_dyn_tg)
            + "</ul>"
        )

    common_dyn = sorted(dr_dyn & tg_dyn)
    for name in common_dyn:
        d_df = dr.dynamicFields[name]
        t_df = tg.dynamicFields[name]
        d_lines = _pretty_lines(_dynamic_to_json(d_df))
        t_lines = _pretty_lines(_dynamic_to_json(t_df))
        if d_lines != t_lines:
            parts.append(f"<h3>{html_lib.escape(name)}</h3>")
            parts.append(
                hd.make_table(
                    d_lines,
                    t_lines,
                    fromdesc="Drupal dynamicField",
                    todesc="Target dynamicField",
                    context=True,
                    numlines=2,
                )
            )

    # ------------------------
    # CopyFields diffs
    # ------------------------
    parts.append(_h("Copy Fields"))
    dr_cf = set((c.source, c.dest, c.maxChars) for c in dr.copyFields)
    tg_cf = set((c.source, c.dest, c.maxChars) for c in tg.copyFields)
    missing_cf = sorted(dr_cf - tg_cf)
    extra_cf = sorted(tg_cf - dr_cf)

    def _fmt_cf(tup):
        s, d, m = tup
        tail = f", maxChars={m}" if m is not None else ""
        return f"{html_lib.escape(s)} → {html_lib.escape(d)}{html_lib.escape(tail)}"

    if missing_cf:
        parts.append(
            "<h3>Missing on Target (will be created)</h3><ul>"
            + "".join(f"<li>{_fmt_cf(t)}</li>" for t in missing_cf)
            + "</ul>"
        )
    if extra_cf:
        parts.append(
            "<h3>Only on Target (left unchanged)</h3><ul>"
            + "".join(f"<li>{_fmt_cf(t)}</li>" for t in extra_cf)
            + "</ul>"
        )

    parts.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# -----------------------------
# Commands
# -----------------------------


@app.command()
def audit(
    drupal_config: str = typer.Option(
        ..., help="Path to directory with schema.xml + extras"
    ),
    target_url: str = typer.Option(
        ...,
        help="SearchStax collection base OR full update URL, e.g., https://HOST/solr/COLLECTION or https://HOST/solr/COLLECTION/update",
    ),
    api_key: Optional[str] = typer.Option(None, help="API key if required"),
    out: Optional[str] = typer.Option(
        None,
        help="If set, write an HTML side-by-side diff report here, e.g., report.html",
    ),
):
    """Load Drupal & target schema and print a concise summary of would-be changes."""
    dr = load_drupal_schema(drupal_config)
    api = SchemaAPI(target_url, api_key)
    tg = target_to_model(api.get_full_schema())
    plan = build_plan(dr, tg)
    # Summary
    typer.echo("Audit summary:")
    typer.echo(f"  create_fieldTypes: {len(plan['create_fieldTypes'])}")
    typer.echo(f"  replace_fieldTypes: {len(plan['replace_fieldTypes'])}")
    typer.echo(f"  create_fields: {len(plan['create_fields'])}")
    typer.echo(f"  create_dynamicFields: {len(plan['create_dynamicFields'])}")
    typer.echo(f"  add_copyFields: {len(plan['add_copyFields'])}")
    if plan["notes"]:
        typer.echo("Notes:")
        for n in plan["notes"]:
            typer.echo(f"  - {n}")

    if out:
        generate_html_report(dr, tg, plan, out)
        typer.echo(f"Wrote HTML diff report to {out}")


@app.command()
def plan(
    drupal_config: str = typer.Option(
        ..., help="Path to directory with schema.xml + extras"
    ),
    target_url: str = typer.Option(
        ...,
        help="SearchStax collection base OR full update URL, e.g., https://HOST/solr/COLLECTION or https://HOST/solr/COLLECTION/update",
    ),
    api_key: Optional[str] = typer.Option(None, help="API key if required"),
    out: str = typer.Option("plan.yaml", help="Where to write the plan YAML"),
):
    """Generate an additive plan (creates + safe fieldType replace) and save to YAML."""
    dr = load_drupal_schema(drupal_config)
    api = SchemaAPI(target_url, api_key)
    tg = target_to_model(api.get_full_schema())
    plan_obj = build_plan(dr, tg)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(plan_obj, f, sort_keys=False)
    typer.echo(f"Wrote plan to {out}")


@app.command()
def apply(
    plan: str = typer.Option(..., help="Path to plan.yaml created by 'plan' command"),
    target_url: str = typer.Option(
        ...,
        help="SearchStax collection base OR full update URL, e.g., https://HOST/solr/COLLECTION or https://HOST/solr/COLLECTION/update",
    ),
    api_key: Optional[str] = typer.Option(None, help="API key if required"),
    yes: bool = typer.Option(False, help="Actually POST changes; otherwise dry-run"),
):
    """Apply the plan to the target Schema API. Dry-run by default (prints what it would send)."""
    with open(plan, "r", encoding="utf-8") as f:
        plan_obj = yaml.safe_load(f)

    api = SchemaAPI(target_url, api_key)

    def _post(payload: Dict[str, Any], label: str):
        pretty = json.dumps(payload, indent=2)
        if not yes:
            typer.echo(f"DRY-RUN {label} → would POST:\n{pretty}\n")
            return
        typer.echo(f"POST {label} ...")
        api.post(payload)
        typer.echo("  OK")

    # 1) FieldTypes (creates)
    for item in plan_obj.get("create_fieldTypes", []):
        _post(item, "add-field-type")

    # 2) FieldTypes (replaces)
    for item in plan_obj.get("replace_fieldTypes", []):
        _post(item, "replace-field-type")

    # 3) Fields
    for item in plan_obj.get("create_fields", []):
        _post(item, "add-field")

    # 4) DynamicFields
    for item in plan_obj.get("create_dynamicFields", []):
        _post(item, "add-dynamic-field")

    # 5) CopyFields
    for item in plan_obj.get("add_copyFields", []):
        _post(item, "add-copy-field")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
