# Release procedure (`v0.1.x → v0.1.{x+1}`)

Each public release mints (a) a tagged git commit, (b) a GitHub Release with the 77 MB picks CSV attached, and (c) a Zenodo software DOI via the `tso1257771/RoSE` ↔ Zenodo integration. Steps:

1. **Pre-flight gate**
   - `pytest tests/ -q` green; `python -m ruff check rose/ tests/ phase_picking/` clean.
   - Fill or refresh placeholder fields:
     - `CITATION.cff` → `authors[0].family-names` / `given-names` / `orcid`, `date-released:`, `version:`.
     - `.zenodo.json` → `creators[0].name` / `affiliation` / `orcid`.
   - If a paper / dataset DOI is now known, uncomment the `references:` block in `CITATION.cff` and add a matching `related_identifiers` entry in `.zenodo.json` (`relation: isSourceOf`, `scheme: doi`).
   - Bump `version` in `CITATION.cff` and (optionally) in `pyproject.toml`.

2. **Tag + push**
   ```bash
   git tag -a vX.Y.Z -m "RoSE toolkit vX.Y.Z"
   git push origin vX.Y.Z
   ```
   Use annotated tags (`-a`) — lightweight tags don't carry metadata that Zenodo / GitHub Releases want.

3. **GitHub Release**
   - Web UI → Releases → "Draft a new release" → pick the tag.
   - Title: `vX.Y.Z — <short description>`. Body: a few lines paraphrasing the changelog since the previous tag.
   - Drag-and-drop `data/Enhanced_ROMPLUS_picks.csv` (77 MB) into the binary-asset area. Optionally add a `.sha256` sidecar.
   - Click "Publish release".

4. **Zenodo software DOI** (auto, if GitHub→Zenodo integration is on at `zenodo.org/account/settings/github/`)
   - The release event triggers Zenodo to create a draft record using `.zenodo.json`.
   - Open Zenodo → Uploads → review the draft (title, creators, license, related identifiers). Edit anything still wrong, then click "Publish".
   - Copy the new DOI. It becomes immutable.

5. **Cross-reference (only when both data + code DOIs exist)**
   - Edit each Zenodo record's *Related identifiers* field:
     - Software record adds: `IsSourceOf: <data DOI>`, `IsDocumentedBy: <paper DOI>`.
     - Data record adds: `IsDocumentedBy: <software DOI>`, `IsCitedBy: <paper DOI>`.
   - In the next minor doc commit, paste both DOIs into the README "Citation" section and uncomment the `references:` block in `CITATION.cff`.

The toolkit can ship a software DOI before the data Zenodo record exists. The cross-references go in after-the-fact; both records remain stable, only metadata updates.

---

## What the Zenodo record actually contains

As of v0.1.0 the record holds exactly one file, the GitHub-generated source
archive `tso1257771/RoSE-v0.1.0.zip` (8.88 MB). Nothing was uploaded by hand,
so the automatic route reproduces the record faithfully. Two consequences:

* **GitHub Release *assets* do not reach Zenodo.** The integration archives the
  repository source only. The 77 MB `Enhanced_ROMPLUS_picks.csv` attached in
  step 3 lives on GitHub alone. To give it a DOI, add it to the Zenodo draft by
  hand in step 4 before publishing.
* Everything committed to the repository, including
  `data/Enhanced_ROMPLUS_catalog.csv`, is inside the archive automatically.

## Concept DOI vs version DOI

Zenodo mints two DOIs. The **concept DOI** `10.5281/zenodo.20250669` always
resolves to the newest version. Each release also gets a **version DOI**
(v0.1.0 is `10.5281/zenodo.20250670`).

Cite the concept DOI in prose and name the version used, so the citation does
not pin readers to an old release. Cite a version DOI only when reproducing a
specific result. `CITATION.cff` carries the concept DOI as `doi:` and lists the
version DOIs under `identifiers:`; add each new version DOI there after
publishing.

## If the GitHub integration is off, or you need extra files

Check the webhook first at <https://zenodo.org/account/settings/github/> — the
`tso1257771/RoSE` toggle must be on *before* the release is published, since
Zenodo only reacts to release events it was listening for.

Otherwise create the new version by hand. On the record page press **New
version**, upload the files, edit the metadata, press **Publish**. The same
thing over the API, with a token from
<https://zenodo.org/account/settings/applications/tokens/new/> carrying the
`deposit:write` and `deposit:actions` scopes:

```bash
TOKEN=...            # never commit this
REC=20250670         # any published version id in the series

# 1. open a new version (returns the new draft id)
NEW=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  "https://zenodo.org/api/deposit/depositions/$REC/actions/newversion" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['links']['latest_draft'].rsplit('/',1)[1])")

# 2. the draft inherits the old files; remove them, then upload the new ones
BUCKET=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "https://zenodo.org/api/deposit/depositions/$NEW" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['links']['bucket'])")
curl -s -X PUT -H "Authorization: Bearer $TOKEN" \
  --upload-file RoSE-v0.2.0.zip "$BUCKET/RoSE-v0.2.0.zip"

# 3. set the version, then publish
curl -s -X PUT -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"metadata":{"version":"v0.2.0"}}' \
  "https://zenodo.org/api/deposit/depositions/$NEW"
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  "https://zenodo.org/api/deposit/depositions/$NEW/actions/publish"
```

Publishing is irreversible: files in a published version can never be changed,
only superseded by a further version.

## Scope of this record

This record is the **toolkit**: the code, the bundled pickers, and the event
catalog. It is not the waveform dataset, and it is not the ROMPLUS bulletin.
Keep the title and description saying so, so it is never mistaken for a
competing release of the source catalog.
