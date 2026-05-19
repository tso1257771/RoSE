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
