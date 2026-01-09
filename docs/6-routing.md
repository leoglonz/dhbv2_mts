# Routing Runoff Simulations

After runoff simulations have been obtained from δHBV 2.0, there are a few options for routing, or accumulating flow, through the river network.

The network can be defined in several ways, with [MERIT-hydro](https://www.reachhydro.org/home/params/merit-basins) and the [NextGen HydroFabric](https://github.com/NOAA-OWP/hydrofabric) being notable examples. In the context of NextGen (ngen), we demonstrate routing on the HydroFabric v2.2 (download [v2.2 source](https://www.lynker-spatial.com) or [AWI-patched v2.2](https://github.com/CIROH-UA/community_hf_patcher/tree/main)) river network with ~4km resolution.

## T-Route (Coming Soon)

[T-Route](https://github.com/NOAA-OWP/t-route) is the standard routing package shipped with [ngen](https://github.com/NOAA-OWP/ngen) and is installed by default when [building ngen with Docker](./4-run_ngen.md/#). This includes support for e.g., Muskingum-Cunge (MC) and diffusive wave routing methods.

For the purposes of this module, we only demonstrate usage of troute within ngen as a post-processor. If you wish to do routing standalone, please see the repo's [official documentation](https://github.com/NOAA-OWP/t-route/blob/master/readme.md).

### T-Route Example

To run e.g. MC routing for our example catchment `cat-2453` in ngen,

```bash
# ngen

# or with Docker (recommended)

```

## Distributed Differentiable Routing (DDR)

Another option for routing is to take the approach applied to HBV here: make the model differentiable and parameterize intelligently with a neural network.

T-route uses parameterizations which are static in time and are not guaranteed to avoid the issue of equifinality: the set of optimal parameters is non-unique and spatially incoherent.

Tadd Bindas et. al (2025, in prep) have shown that there is value in generalization, physical consistency, and streamflow forecast performance by leveraging big data to parameterize the Muskingum-Cunge (MC) model. We call this differentiable MC, denoted by δMC, named after the requirement that the MC model itself be differntiable to support gradient-based training of the neural network.

This routing method has been demonstrated alongside naive hydrograph routing methods to improve skill for δHBV 2.0 (**sources**).

[Distributed Differentiable Routing (DDR)](https://github.com/DeepGroundwater/ddr) is an operational formalization of this philosophy which currently supports δMC. While there are plans to support DDR within ngen in some capacity, routing must be done separately on ngen's runoff outputs.

### DDR Setup

#### (1) Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/leoglonz/ddr.git
    cd ddr
    ```

2. Optional - Create a conda/venv environment to isolate dependencies:

    DDR requires `>=Py3.11`, so a new env may be required if dhbv2 was previously installed for ngen.

    ```bash
    conda create -n ddr python=3.11
    conda activate ddr

    # or (recommended)

    uv venv --python=3.11 .venv_ddr
    source .venv_ddr/bin/activate
    ```

3. Install dependencies:

    We recommend [Astral UV](https://docs.astral.sh/uv/) to install packages (available via `pip install uv`), however standard `pip install` will also work.

    ```bash
    cd ./ddr
    uv pip install . ./engine

    # or in editable mode
    uv pip install -e . -e ./engine
    ```

#### (2) Data

...

### DDR Example

To run δMC routing for our example catchments `cat-2453`, `cat-2454`, `cat-2455`

```bash



```
