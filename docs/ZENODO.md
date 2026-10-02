# Depositing RoSE: two records

RoSE is deposited as two Zenodo records that cite each other and can be used
apart.

| Record | Holds | Type | Licence | Size |
|---|---|---|---|---|
| **Dataset** | the waveforms, the trace metadata, the event catalog, the picks, the instrument responses | dataset | CC-BY-4.0 | about 35 GB |
| **Toolkit** | this repository at a tagged release, which carries the loader, the pickers, the magnitude calibration and the drivers | software | MIT | a few MB |

The split is deliberate. Somebody who wants the waveforms should not download
a Python package to get them, and somebody who wants to check a magnitude
should not download 35 GB to read a coefficient table. The toolkit record
already exists under concept DOI
[10.5281/zenodo.20250669](https://doi.org/10.5281/zenodo.20250669). The
dataset record is new.

[`RELEASING.md`](../RELEASING.md) covers the mechanics of a toolkit release.
This file says what goes in each record and in what order.

## Before anything: the deposited metadata is out of date

**The archive built before the magnitudes were released must not be
deposited.** Its `metadata{YEAR}.csv` files carry the NIEP bulletin local
magnitude in `source_magnitude`, with `source_magnitude_type` reading `ml` for
every trace. Of the 19,228 events in that build, `source_magnitude` matches
`ML_ROMPLUS` for 14,162 and the released `ML` for 202. The build also has no
`split` column and none of the eleven released magnitude fields.

Depositing it would mint a permanent DOI on the one magnitude the data
descriptor argues against using, and anyone training a magnitude model on it
would be fitting the bulletin.

The waveforms are not affected. Only the metadata tables are, so they can be
rewritten in place:

```bash
python tools/repair_archive_metadata.py /path/to/data/rose --check   # report only
python tools/repair_archive_metadata.py /path/to/data/rose
```

That adds the eleven magnitude fields, sets `source_magnitude` to Mw where it
was measured and ML otherwise, sets `source_magnitude_type` to the scale
actually stored, and regenerates the deterministic `split` column. On the
present build it puts 219,002 of 219,272 traces on Mw, 228 on ML and 42 on
neither. It rewrites a few hundred megabytes of CSV and leaves the 35 GB of
HDF5 untouched.

Check the result against what the manuscript claims before going further:

```bash
head -1 /path/to/data/rose/metadata2014.csv | tr ',' '\n' | grep -E 'split|source_mw|source_ml'
```

## Record 1: the dataset

Zenodo takes 50 GB per record, so the archive fits with room to spare.

| File | What it is |
|---|---|
| `waveforms{2014..2024}.hdf5` | the waveforms, 11 files, about 35 GB, ZNE order, 100 Hz, counts |
| `metadata{2014..2024}.csv` | one row per trace, the schema in [`SEISBENCH_FORMAT.md`](SEISBENCH_FORMAT.md) |
| `chunks` | the chunk list SeisBench reads to find the yearly files |
| `Enhanced_ROMPLUS_catalog.csv` | 19,230 events with the released Mw and ML |
| `Enhanced_ROMPLUS_picks.csv` | 416,063 picks, about 74 MB |
| `rose_stationxml.zip` | 212 StationXML files, the instrument responses |
| `SHA256SUMS` | a checksum per file |
| `README.md` | what the record is, how to load it, what to cite |

Two of these are not where they should be today. The picks table is a GitHub
release asset, which is neither persistent nor citable, and the responses are
needed to go from counts to ground motion, so both belong in the record.

Build the checksums from the archive directory:

```bash
cd /path/to/data
( find rose -type f \( -name '*.hdf5' -o -name '*.csv' -o -name chunks \) | sort | xargs sha256sum
  sha256sum Enhanced_ROMPLUS_catalog.csv Enhanced_ROMPLUS_picks.csv rose_stationxml.zip
) > SHA256SUMS
sha256sum -c SHA256SUMS      # and again after the upload, from the downloaded copy
```

Metadata for the record is in [`zenodo-dataset.json`](../zenodo-dataset.json).
Upload it with the Zenodo API or paste the fields in by hand. Set
`upload_type` to `dataset` and the licence to CC-BY-4.0, which is the licence
of the waveforms and of the derived catalog. The ROMPLUS bulletin behind them
is a registered NIEP product and is cited separately, so say so in the
description rather than relicensing it.

## Record 2: the toolkit

The GitHub integration archives the repository source on every tagged
release, so the record is the repository at that tag, which is what the code
DOI should mean. Tag, and the draft appears:

```bash
git tag -a v0.3.0 -m "RoSE 0.3.0: the released magnitudes as a reusable package"
git push origin v0.3.0
```

Then, in the Zenodo draft, before publishing:

1. Attach `Enhanced_ROMPLUS_picks.csv` if it is not yet in the dataset record.
   Release assets do not reach Zenodo. The integration takes the source
   archive only.
2. Add the dataset DOI under related identifiers as `isDocumentedBy`, once
   record 1 is published.
3. Check the version field matches the tag.

[`.zenodo.json`](../.zenodo.json) supplies the rest.

## Order of operations

The two records point at each other, which cannot be done in one pass. Zenodo
reserves a DOI for a draft before it is published, so the reservation is what
breaks the circle.

1. Repair the archive metadata, then build `SHA256SUMS`.
2. Create the dataset draft, reserve its DOI, upload the files. Do not publish.
3. Put the reserved dataset DOI into `.zenodo.json`, `CITATION.cff` and the
   manuscript's data availability section. Commit.
4. Tag the toolkit release and let the integration build its draft. Reserve
   its DOI too.
5. Cross-reference: the dataset record cites the toolkit DOI as
   `isSupplementedBy`, the toolkit record cites the dataset DOI as
   `isDocumentedBy`, and both cite the paper once it has a DOI.
6. Publish the dataset record, then the toolkit record.
7. Verify the checksums from a fresh download, and that
   `pip install "rose-seismic @ git+https://github.com/tso1257771/RoSE@v0.3.0"`
   works in a clean environment.

Both records should be given reviewer access before submission. ESSD asks for
the data to be deposited with a DOI at submission, and a referee who cannot
open the archive will say so.

## What still has to be decided

* The dataset record needs an author list. The toolkit record has one author,
  but the waveforms are a NIEP product and the relocations and magnitudes are
  this work, so the dataset authorship is not the same list and is not ours
  alone to set.
* The redistribution terms of the ROMPLUS bulletin have to be stated. "Access
  is open" is not a licence, and the manuscript currently names none for the
  data.
