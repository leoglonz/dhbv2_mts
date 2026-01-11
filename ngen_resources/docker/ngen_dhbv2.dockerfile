# Adaptation of CIROH-UA/NGIAB-CloudInfra

# Stage 1: Base image with dependencies
FROM rockylinux:9.1 AS base
ARG NPROC=8

ENV NGEN_REPO=mhpi/ngen
ENV NGEN_BRANCH=master
# NOAA-OWP/t-route needs work to be compatible here
ENV TROUTE_REPO=CIROH-UA/t-route
ENV TROUTE_BRANCH=ngiab

RUN echo "max_parallel_downloads=10" >> /etc/dnf/dnf.conf
RUN dnf update -y && \
    dnf install -y epel-release && \
    dnf config-manager --set-enabled crb && \
    dnf install -y \
    vim libgfortran sqlite \
    bzip2 expat udunits2 zlib \
    mpich hdf5 netcdf netcdf-fortran netcdf-cxx netcdf-cxx4-mpich

# Install astral-uv
ENV PATH="/root/.cargo/bin:${PATH}"
ENV UV_INSTALL_DIR=/root/.cargo/bin
ENV UV_COMPILE_BYTECODE=1
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv self update


# Stage 2: Build base with development tools
FROM base AS build_base

RUN echo "max_parallel_downloads=10" >> /etc/dnf/dnf.conf
RUN dnf install -y epel-release && \
    dnf config-manager --set-enabled crb && \
    dnf install -y \
    sudo gcc gcc-c++ make cmake ninja-build tar git gcc-gfortran libgfortran sqlite sqlite-devel \
    python3 python3-devel python3-pip \
    expat-devel flex bison udunits2-devel zlib-devel \
    wget mpich-devel hdf5-devel netcdf-devel \
    netcdf-fortran-devel netcdf-cxx-devel lld clang


# Stage 3: Boost setup
FROM build_base AS boost_setup

RUN curl -L -o boost_1_86_0.tar.bz2 https://sourceforge.net/projects/boost/files/boost/1.86.0/boost_1_86_0.tar.bz2/download \
    && tar -xjf boost_1_86_0.tar.bz2 \
    && rm boost_1_86_0.tar.bz2
ENV BOOST_ROOT=/boost_1_86_0


# Stage 4: Install T-Route dependencies
FROM boost_setup AS troute_prebuild

WORKDIR /ngen

# Troute looks for netcdf.mod in the wrong place unless we set this
ENV FC=gfortran NETCDF=/usr/lib64/gfortran/modules/
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /ngen/
RUN uv venv
ENV PATH="/ngen/.venv/bin:$PATH"

# make sure clone isn't cached if repo is updated
ADD https://api.github.com/repos/${TROUTE_REPO}/git/refs/heads/${TROUTE_BRANCH} /tmp/version.json

RUN uv pip install -r https://raw.githubusercontent.com/$TROUTE_REPO/refs/heads/$TROUTE_BRANCH/requirements.txt


# Stage 5: Build T-Route
FROM troute_prebuild AS troute_build

WORKDIR /ngen/t-route

# Clone and init all submodules
RUN git clone --depth 1 --single-branch -b $TROUTE_BRANCH https://github.com/$TROUTE_REPO.git . \
    && git submodule update --init --recursive --jobs ${NPROC} --depth 1 \
    && uv pip install build wheel

# disable everything except the kernel builds
RUN sed -i 's/build_[a-z]*=/#&/' compiler.sh

RUN ./compiler.sh no-e

# install / build using UV because it's so much faster
# no build isolation needed because of cython namespace issues
RUN uv pip install --config-setting='--build-option=--use-cython' src/troute-network/
RUN uv build --wheel --config-setting='--build-option=--use-cython' src/troute-network/
RUN uv pip install --no-build-isolation --config-setting='--build-option=--use-cython' src/troute-routing/
RUN uv build --wheel --no-build-isolation --config-setting='--build-option=--use-cython' src/troute-routing/
RUN uv build --wheel --no-build-isolation src/troute-config/
RUN uv build --wheel --no-build-isolation src/troute-nwm/


# Stage 6: Clone ngen
FROM troute_prebuild AS ngen_clone

WORKDIR /ngen

# Clone and init all submodules
ADD https://api.github.com/repos/${NGEN_REPO}/git/refs/heads/${NGEN_BRANCH} /tmp/version.json
RUN git clone --recursive --single-branch -b $NGEN_BRANCH https://github.com/$NGEN_REPO.git \
    && cd ngen \
    && git submodule update --init --recursive --jobs ${NPROC} --depth 1 \
    && git submodule status


# Stage 7: Build ngen
FROM ngen_clone AS ngen_build

ENV VIRTUAL_ENV=/ngen/.venv
ENV PATH="/ngen/.venv/bin:$PATH"
ENV PATH=${PATH}:/usr/lib64/mpich/bin

WORKDIR /ngen/ngen

# Pin numpy to <2.0.0 to avoid issues with submodules
RUN uv pip install "numpy<2.0.0"

ARG COMMON_BUILD_ARGS=" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=. \
    \
    -DNGEN_WITH_NETCDF:BOOL=ON \
    -DNGEN_WITH_SQLITE:BOOL=ON \
    -DNGEN_WITH_UDUNITS:BOOL=ON \
    -DNGEN_WITH_BMI_FORTRAN:BOOL=ON \
    -DNGEN_WITH_BMI_C:BOOL=ON \
    -DNGEN_WITH_PYTHON:BOOL=ON \
    -DNGEN_WITH_ROUTING:BOOL=ON \
    -DNGEN_WITH_TESTS:BOOL=ON \
    -DNGEN_QUIET:BOOL=OFF \
    -DUDUNITS_QUIET:BOOL=ON \
    \
    -DNGEN_WITH_EXTERN_SLOTH:BOOL=ON \
    -DNGEN_WITH_EXTERN_TOPMODEL:BOOL=ON \
    -DNGEN_WITH_EXTERN_CFE:BOOL=ON \
    -DNGEN_WITH_EXTERN_PET:BOOL=ON \
    -DNGEN_WITH_EXTERN_NOAH_OWP_MODULAR:BOOL=ON"

RUN cmake \
    -G Ninja \
    -B build_serial \
    -S . \
    ${COMMON_BUILD_ARGS} \
    -DNGEN_WITH_MPI:BOOL=OFF


RUN cmake \
    --build build_serial \
    --target all \
    -- \
    -j ${NPROC}

ARG MPI_BUILD_ARGS=" \
    -DNGEN_WITH_MPI:BOOL=ON \
    -DNetCDF_ROOT=/usr/lib64/mpich \
    -DCMAKE_PREFIX_PATH=/usr/lib64/mpich \
    -DCMAKE_LIBRARY_PATH=/usr/lib64/mpich/lib"

# Install the mpi enabled netcdf library and build Ngen parallel with it
RUN dnf install -y netcdf-cxx4-mpich-devel
RUN cmake \
    -G Ninja \
    -B build_parallel \
    -S . ${COMMON_BUILD_ARGS} ${MPI_BUILD_ARGS} \
    -DNetCDF_CXX_INCLUDE_DIR=/usr/include/mpich-$(arch) \
    -DNetCDF_INCLUDE_DIR=/usr/include/mpich-$(arch)

RUN cmake \
    --build build_parallel \
    --target all \
    -- \
    -j ${NPROC}


# Stage 8: Restructure files for final image
FROM ngen_build AS restructure_files

# Setup final directories and permissions
RUN mkdir -p /dmod/datasets /dmod/datasets/static /dmod/shared_libs /dmod/bin /dmod/utils/ \
    && shopt -s globstar \
    && cp -a ./extern/**/cmake_build/*.so* /dmod/shared_libs/. || true \
    && cp -a ./extern/noah-owp-modular/**/*.TBL /dmod/datasets/static \
    && cp -a ./build_parallel/ngen /dmod/bin/ngen-parallel || true \
    && cp -a ./build_serial/ngen /dmod/bin/ngen-serial || true \
    && cp -a ./build_parallel/partitionGenerator /dmod/bin/partitionGenerator || true \
    && cp -ar ./utilities/* /dmod/utils/ \
    && cd /dmod/bin \
    && (stat ngen-parallel && ln -s ngen-parallel ngen) || (stat ngen-serial \
    && ln -s ngen-serial ngen)


# Stage 9: Development image
FROM restructure_files AS dev

# Set up library path
RUN echo "/dmod/shared_libs/" >> /etc/ld.so.conf.d/ngen.conf && ldconfig -v

# Add mpirun to path
ENV PATH=$PATH:/usr/lib64/mpich/bin

#  Set permissions
RUN chmod a+x /dmod/bin/*
RUN mv /ngen/ngen /ngen/ngen_src
WORKDIR /ngen


# Stage 10: Final image
FROM base AS final

WORKDIR /ngen

# Copy necessary files from build stages
COPY --from=restructure_files /dmod /dmod
COPY --from=troute_build /ngen/t-route/src/troute-*/dist/*.whl /tmp/

RUN ln -s /dmod/bin/ngen /usr/local/bin/ngen

RUN uv venv \
    && uv pip install --no-cache-dir /tmp/*.whl netCDF4==1.6.3
RUN rm -rf /tmp/*.whl

# Set up library path
RUN echo "/dmod/shared_libs/" >> /etc/ld.so.conf.d/ngen.conf && ldconfig -v

# Add mpirun to path
ENV PATH=$PATH:/usr/lib64/mpich/bin

RUN uv pip install numpy==$(/dmod/bin/ngen --info | grep -e 'NumPy Version: ' | cut -d ':' -f 2 | uniq | xargs)

# now that the only version of numpy is the one that NGen expects,
# we can add the venv to the path so ngen can find it
ENV PATH="/ngen/.venv/bin:${PATH}"
ENV CI=true

# Install dhbv and model weights
COPY --from=ngen_clone /ngen/ngen/extern/dhbv2/dhbv2 /ngen/ngen/extern/dhbv2
RUN dnf -y install git gcc python3-devel gcc-c++ awscli unzip

RUN uv pip install /ngen/ngen/extern/dhbv2 \
    && aws s3 cp s3://mhpi-spatial/mhpi-release/models/owp/dhbv_2_mts.zip ./temp/ --no-sign-request \
    && unzip ./temp/dhbv_2_mts.zip -d ./temp \
    && mv ./temp/dhbv_2_mts/ /ngen/ngen/extern/dhbv2/ngen_resources/data/dhbv_2_mts/model/ \
    && rm -r ./temp


RUN echo "export PS1='\u\[\033[01;32m\]@ngiab_dev\[\033[00m\]:\[\033[01;35m\]\W\[\033[00m\]\$ '" >> ~/.bashrc

# Suppress warning from troute + dmod (?)
ENV PYTHONWARNINGS="ignore:networkx backend defined more than once:RuntimeWarning"

CMD ["./ngen", "data/catchment_data.geojson", "", "data/nexus_data.geojson", "", "data/example_realization_config.json"]
