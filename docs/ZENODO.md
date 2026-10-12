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
