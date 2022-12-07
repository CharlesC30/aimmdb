# AIMM Schemas

This document describes the schemas enforced within aimmdb.
Schema enforcement is done through tiled's [spec](https://blueskyproject.io/tiled/explanations/metadata.html#specs) mechanism.
Each record in the database is given one or more specs which are strings labeling schemas the record conforms to.
Each spec is associated with a validation function which checks the schema on upload.
Each dataset in aimmdb defines a list of specs that it will accept.

The validation functions can check the `metadata`, `structure_family`, and `structure` (see tiled documentation) of uploaded records but NOT the data itself.

## Metadata schemas

The validation functions defined in aimmdb make use of metadata schemas defined
in [pydantic](https://pydantic-docs.helpmanual.io/) to validate the metadata
(additional validation may be performed on the `structure_family` and `structure`
to e.g. ensure that certain columns are present in a dataframe).
Below we list the main metadata schemas with an informal table representation.
To understand the details see the [code](https://github.com/AI-multimodal/aimmdb/blob/dev/aimmdb/schemas.py).

* ExperimentalXASMetadata

    Generic metadata for experimental XAS measurements.
    Adapted from [XDI](https://github.com/XraySpectroscopy/XAS-Data-Interchange) format.

    | Key | Description |
    | --- | --- |
    | dataset | aimmdb dataset for record |
    | sample_id | id of sample in sample database (optional) |
    | element.edge | xray absorption edge (K,L2,L3, etc...) | 
    | element.symbol | absorbing atom |
    | facility.name | name of the facility (NSLSII, ALS, APS, etc...) |
    | beamline.name | name of the beamline |

* BatteryChargeMetadata

    Metadata for measurements of battery materials.

    | Key | Description |
    | --- | --- |
    | charge.cycle | charge cycle |
    | charge.voltage | battery voltage |
    | charge.state | charge state (C = charged, DC = discharged) | 



## Specs

Specs are associated with a validation function which enforces a schema on upload.
Below we list the defined specs together with a description of their requirements.
NOTE: a single record may have more than one spec (for example a dataframe with
`i0` `tfy` and `tey` columns may have both `XAS_TFY` and `XAS_TEY` specs).

* XAS_TFY

    Total fluoresence yield xas measurement.
    - `structure_family` must be dataframe
    - `structure` must contain `energy`, `i0`, and `tfy` columns
    - `metadata` must conform to requirements of `ExperimentalXASMetadata`

* XAS_TEY

    Total electron yield xas measurement.
    - `structure_family` must be dataframe
    - `structure` must contain `energy`, `i0`, and `tey` columns
    - `metadata` must conform to requirements of `ExperimentalXASMetadata`

* XAS_trans

    Transmission mode xas measurement
    - `structure_family` must be dataframe
    - `structure` must contain `energy`, `i0`, and `itrans` columns
    - `metadata` must conform to requirements of `ExperimentalXASMetadata`

* HasBatteryChargeData

    Any kind of measurement of a battery material which contains metadata about the battery's charge state.
    - `metadata` must conform to requirements of `BatteryChargeMetadata`.

## Glossary

Below we provide a brief glossary of terms appearing in the above schemas.
For an in depth description of experimental XAS techniques and terms see a more detailed reference (e.g. [[1](https://www.lehigh.edu/imi/teched/GlassCSC/SuppReading/Tutorials.pdf)]).

* energy: incident beam photon energy.
* i0: incident beam flux.
* itrans: transmitted beam flux in transmission mode experiment. Absorption cross section estimated by $\mu(E) \sim \log(I_0 / I_{\mathrm{trans}})$.
* tfy: total fluorescence yield. Measures the flux of photons emitted from the decay of the excited state. Absorption cross section estimated by $\mu(E) = \mathrm{tfy} / I_0$. Surface sensitive (~100-200 nm).
* tey: total electron yield. Measures the flux of electrons emitted by Auger effect. Absorption cross section estimated by $\mu(E) = \mathrm{tey} / I_0$. Surface sensitive (~10 nm).
