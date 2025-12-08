To use in the NextGen Framework,
1) Copy contents of the dhbv2 repo into `ngen/extern/dhbv2/dhbv2/`.
2) Copy contents of `ngen_resources/data/` into `./ngen/data/`. This contains data for the model, realizations for NextGen, and other config files enabling the dhbv2 package to run from within NextGen.

`ngen_resources/data/` contains:
- δHBV2.0 model, BMI, and routing configuration files in `config/`,
- Pretrained model weights for δHBV2.0 in `models/`,
- "Realization" configuration files for NextGen in `realization/`,
- CONUS-scale statistics for static catchment attributes, catchment and nexus data GeoJSON files, and a subset (Juniata River Basin) of the NextGen hydrofabric v2.2 in `spatial/`,
- AORC forcing data for NextGen + δHBV2.0 forward inference on the Juniata River Basin in `forcing/`.
