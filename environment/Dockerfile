# hash:sha256:cd6e5d3fdc0fc04cd450fef07f1ecfe02217d15ee4fa5d4ce575e7bc50da1533
FROM registry.codeocean.com/codeocean/tensorflow:1.4.0-python3.5.2-cuda8.0.61-cudnn6.0.21-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gdal-bin=1.11.3+dfsg-3build2 \
        python-gdal=1.11.3+dfsg-3build2 \
        python3-gdal=1.11.3+dfsg-3build2 \
        python3-rtree=0.8.2+ds-2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    argparse==1.1 \
    efficientnet==1.0.0 \
    fiona==1.8.4 \
    geopandas==0.4.1 \
    joblib==0.11 \
    keras==2.2.4 \
    numpy==1.16.2 \
    opencv-python==4.1.2.30 \
    pandas==0.24.2 \
    rasterio==1.0.15 \
    rasterstats==0.13.1 \
    scikit-image==0.15.0 \
    scikit-learn==0.20.3 \
    scipy==1.2.1 \
    seaborn==0.9.0 \
    shapely==1.6.4 \
    tensorflow-gpu==1.8.0 \
    tqdm==4.30.0
