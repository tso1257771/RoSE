# Zenodo records

RoSE uses separate records for software and data because their rights differ.

## Software record

The software record may contain the source code, tests, documentation,
calibration tables produced by the authors, and model files whose licenses
allow redistribution. It must not contain the compiled event catalog, phase
pick tables, waveform files, or StationXML files unless their redistribution
rights have been documented.

The GitHub to Zenodo integration creates a source archive from each tagged
release. Check the archive before publication because every tracked file is
included. Do not attach data files to the GitHub Release or the Zenodo software
record.

The archived v0.1.0 and v0.3.0 software records are temporarily restricted
because their original source packages contained the compiled event catalog.
The current GitHub branch and release tags no longer contain that file. Publish
a corrected code-only Zenodo version only after checking the generated source
archive.

## Data record

The data record is restricted while the authors confirm the rights for the
following material:

- NIEP event fields for 2023 and 2024
- analyst phase picks
- waveform files
- StationXML and instrument response metadata
- products that combine these materials with results produced for RoSE

The CC BY 4.0 license for ROMPLUS 984--2022 covers the event catalog in that
release. It does not cover phase picks, waveforms, response metadata, or later
NIEP products. Files received from a collaborator may be published only when
the collaborator or the rights holder confirms the redistribution terms in
writing.

## Data record versions

The concept DOI [`10.5281/zenodo.23153183`](https://doi.org/10.5281/zenodo.23153183)
always resolves to the newest version. Zenodo lists the versions without
version names:

| Version | DOI | Published | Changes |
|---|---|---|---|
| 1 | [`10.5281/zenodo.23153184`](https://doi.org/10.5281/zenodo.23153184) | 2026-10-05 | earthquake traces (chunks `2014` … `2024`), event catalog, picks and StationXML |
| 2 | [`10.5281/zenodo.23495039`](https://doi.org/10.5281/zenodo.23495039) | 2026-10-20 | adds the noise chunks `noise2014` … `noise2024`; the earthquake metadata leave missing picks empty instead of `-1` and give every time stamp fractional seconds. Its noise metadata do not match its noise waveforms, so do not use its noise chunks |
| 3 | [`10.5281/zenodo.23681945`](https://doi.org/10.5281/zenodo.23681945) | 2026-10-27 | replaces the 11 noise metadata files, `waveformsnoise2014.hdf5` and `waveformsnoise2016.hdf5`, so the noise metadata match the noise waveforms; the other 36 files are those of version 2 |

The `chunks` file of versions 2 and 3 lists the year chunks and the noise
chunks. Toolkit v0.4.0 and later load only the earthquake traces unless
`include_noise=True` is passed. Older toolkit versions load every chunk in the
file; pass them `chunks=[str(y) for y in range(2014, 2025)]` to keep the noise
out.

## Release check

Before publishing or reopening a data record:

1. List every file and the fields it contains.
2. Record the source and license for each file.
3. Separate material with different licenses.
4. Confirm that the selected license permits copying, modification,
   redistribution, and commercial use when CC BY 4.0 is used.
5. Keep the written permission with the release records.
6. Check the source archive and all GitHub Release assets before publication.

If raw waveforms cannot be redistributed, publish a station and time manifest
with retrieval code for the provider's FDSN or EIDA service. Do not move the
same unresolved files to another public repository.
