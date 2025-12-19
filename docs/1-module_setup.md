## Loading Model Weights

dhbv2 uses an LSTM which requires loading trained weights. Since these are too large to store with the module, they must be downloaded and installed from AWS S3 as follows. First, navigate to the submodule.

    cd extern/dhbv2/dhbv2

If AWS CLI is not installed on your system (check `aws --version`) see [AWS instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) for system-specific directions.

Then download model weights (and coupled normalization statistics file) from S3 like

    aws s3 cp s3://mhpi-spatial/mhpi-release/models/owp/dhbv_2_hfv2.2_15y_daily.zip . --no-sign-request
    unzip 4-dhbv_2.zip -d /temp
    mv /temp/dhbv_2_hfv2.2_15y_daily/. /ngen_resources/data/dhbv2/models/hfv2.2_15yr
    rm -r /temp

Note: Other models made available will be located in `.dhbv2/models/` with a readme providing the specific S3 URI to use above.

...
