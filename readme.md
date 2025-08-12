# Drupal → SearchStax Site Search Schema Migrator (MVP)

A small Python CLI that helps Drupal users migrate their **Search API Solr** schema (local XML files) into a **SearchStax Site Search** app using the Solr **Schema API**.

**What it does (MVP):**

- Parses local Drupal config: `schema.xml`, `schema_extra_types.xml`, `schema_extra_fields.xml`.
- Fetches the target Site Search schema via HTTP.
- Builds an **additive-only** plan:
  - Creates missing **fieldTypes**, **fields**, **dynamicFields**, **copyFields**.
  - Performs **surgical fieldType updates** when safe: inserts missing analyzer factories (char filters / token filters) in the exact Drupal order **without** removing or reordering existing ones.
  - Supports **both** analyzers per field type: `indexAnalyzer` and `queryAnalyzer`.
- Applies that plan to the target (dry-run by default).

> ✋ The tool never deletes or in-place edits fields/dynamicFields/copyFields. For fieldTypes, it will **only** do a replace when it’s an additive analyzer change with the **same tokenizer**; otherwise it skips with a note.

---

## Requirements

- Python 3.10+
- Install dependencies:

```
pip install -r requirements.txt
```

**Pinned versions** (in `requirements.txt`):

```
typer[all]==0.12.5
click==8.1.7
rich==13.7.1
requests==2.32.3
PyYAML==6.0.2
lxml==5.3.0
```

---

## Authentication & URLs

- **Auth header:** `Authorization: Token <API_KEY>`
- **URL normalization:** you may pass any of these and the tool will normalize to the collection base internally:
  - `https://.../solr/COLLECTION`
  - `https://.../solr/COLLECTION/update`
  - `https://.../solr/COLLECTION/schema`

---

## Quickstart

### 1) Audit (see what would change)

**Bash**

```bash
python drupal_schema_migrator.py audit \
  --drupal-config ./drupal-solr-conf \
  --target-url "https://searchcloud-2-us-west-2.searchstax.com/29847/devtry2-5451/update" \
  --api-key YOUR_API_KEY
```

**PowerShell**

```powershell
python .\drupal_schema_migrator.py audit `
  --drupal-config "C:\\path\\to\\drupal-solr-conf" `
  --target-url "https://searchcloud-2-us-west-2.searchstax.com/29847/devtry2-5451/update" `
  --api-key YOUR_API_KEY
```

### 2) Generate a plan

```bash
python drupal_schema_migrator.py plan \
  --drupal-config ./drupal-solr-conf \
  --target-url "https://searchcloud-2-us-west-2.searchstax.com/29847/devtry2-5451/update" \
  --api-key YOUR_API_KEY \
  --out plan.yaml
```

### 3) Apply the plan (dry-run)

```bash
python drupal_schema_migrator.py apply \
  --plan plan.yaml \
  --target-url "https://searchcloud-2-us-west-2.searchstax.com/29847/devtry2-5451/update" \
  --api-key YOUR_API_KEY
```

### 4) Apply the plan (real changes)

```bash
python drupal_schema_migrator.py apply \
  --plan plan.yaml \
  --target-url "https://searchcloud-2-us-west-2.searchstax.com/29847/devtry2-5451/update" \
  --api-key YOUR_API_KEY \
  --yes
```

---

## How it works

1. **Parse Drupal XML**
   - Loads `schema.xml` (and resolves Drupal’s `&extratypes;` / `&extrafields;` entities) plus optional `schema_extra_types.xml`, `schema_extra_fields.xml`.
2. **Fetch target schema** from the Site Search app via the Schema API.
3. **Diff**
   - **Create** anything missing on target.
   - For **fieldTypes** found on both sides, compare **indexAnalyzer** and **queryAnalyzer** separately. If the **tokenizer matches** and only **additional factories** (char filters or token filters) are missing on target, the tool plans a `replace-field-type` that inserts those factories **in the exact Drupal order**.
4. **Apply**
   - Posts `add-field-type`, `replace-field-type`, `add-field`, `add-dynamic-field`, `add-copy-field` calls in a safe order.

---

## Examples: analyzer merge

Given Drupal `text_en` adds a `PatternReplaceCharFilterFactory` and Site Search lacks it, the tool will:

- Detect identical tokenizers (`WhitespaceTokenizerFactory`).
- Insert the missing `PatternReplaceCharFilterFactory` **after** `MappingCharFilterFactory` (index analyzer) and at the appropriate positions (query analyzer), preserving all existing target filters.
- Emit a single `replace-field-type` with both `indexAnalyzer` and `queryAnalyzer`.

If tokenizers **differ** within an analyzer type, the tool **skips** replacing that fieldType and prints a note (no risky updates).

---

## Troubleshooting

### Click / Typer help error

If `--help` throws `TypeError: Parameter.make_metavar() missing 1 required positional argument: 'ctx'`, your Click is too old. Install pinned requirements or run:

```
pip install --upgrade "click==8.1.7" "rich==13.7.1"
```

### XML entity errors (Drupal includes)

If you see `XMLSyntaxError: Entity 'extrafields' not defined`:

- Ensure `schema_extra_fields.xml` and `schema_extra_types.xml` are in the **same directory** as `schema.xml`.
- The tool now preflights `schema.xml` and will throw a clear error listing any missing include files and the expected paths.

### Auth failures

Confirm you’re using **Token auth**:

```
Authorization: Token <API_KEY>
```

### Nothing created

- The plan is **additive-only**. If names already exist and match, it won’t update them.
- For fieldTypes, only additive analyzer inserts with matching tokenizers are replaced. Otherwise you’ll see a note explaining why it skipped.

---

## Design choices (MVP)

- **Additive-only** for fields/dynamicFields/copyFields — no deletes or in-place mutations.
- **Surgical replace** for fieldTypes **only when safe** (tokenizer match, additive factories).
- **Order-preserving inserts** for analyzer factories.
- **Idempotent**: safe to re-run; it will skip already-applied items.
- **Dry-run by default** for `apply`.

---

## Roadmap

- Configurable rules for parameter augmentation (e.g., add `ignoreCase=true` if missing).
- Shadow-type option when tokenizers differ (e.g., `text_en__drupal`) and automatic remapping for new fields.
- Heuristics/templates for common Drupal suffixes (`*_t`, `*_s`, etc.).
- Better quota checks and rollback guidance.

---

## Contributing

PRs welcome. Please include:

- A minimal Drupal fixture (anonymized)
- Repro steps and expected plan output
- Unit tests for parser and analyzer merge logic

---

## License

Apache-2.0 license

