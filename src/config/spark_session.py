from __future__ import annotations

"""Spark session factory used by delay-data and model training code."""

import os
import pwd
from random import randrange
import sys

from .settings import ProjectSettings, get_settings


def get_spark_session(
    settings: ProjectSettings | None = None,
    app_name: str | None = None,
    master: str | None = None,
):
    """Return a Spark session configured for the COM-490 Iceberg datasets."""
    settings = settings or get_settings()

    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError("PySpark is required for delay-data preparation and model training.") from exc

    username = pwd.getpwuid(os.getuid()).pw_name
    hadoop_fs = settings.hadoop_fs
    app_name = app_name or f"{username}-robust-journey-planner"
    master = master or settings.spark_master or ("yarn" if hadoop_fs else "local[*]")

    # TLJH/venv environments disable user site-packages, so ~/.local is not in
    # sys.path on the driver. Explicitly prepend it so executor nodes (which
    # share home dirs via NFS) can find packages installed with pip install --user.
    python_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    user_site = os.path.expanduser(f"~/.local/lib/{python_ver}/site-packages")
    exec_pythonpath = ":".join(p for p in [user_site] + sys.path if p)

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.ui.port", randrange(4050, 4450, 5))
        .config("spark.executorEnv.PYTHONPATH", exec_pythonpath)
    )

    if hadoop_fs:
        # Deploy the professor's pre-built XGBoost venv archive to all executor nodes.
        # XGBoost is available on the driver (TLJH hub) but not in the executor Python
        # environment by default — this is required for SparkXGBRegressor distributed training.
        os.environ["PYSPARK_PYTHON"] = "./env/xgboost-env/bin/python3"
        builder = (
            builder.config(
                "spark.yarn.dist.archives",
                "hdfs:///data/com-490/python/archives/xgboost-env.tar.gz#env",
            )
        )

        builder = (
            builder.config(
                "spark.jars",
                f"{hadoop_fs}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar,"
                f"{hadoop_fs}/data/com-490/jars/sedona-spark-shaded-3.5_2.13-1.7.1.jar,"
                f"{hadoop_fs}/data/com-490/jars/geotools-wrapper-1.7.1-28.5.jar",
            )
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.iceberg.type", "hadoop")
            .config("spark.sql.catalog.iceberg.warehouse", f"{hadoop_fs}/data/com-490/silver/")
            .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
            .config("spark.sql.catalog.spark_catalog.type", "hadoop")
            .config("spark.sql.catalog.spark_catalog.warehouse", f"{hadoop_fs}/user/{username}/final/warehouse")
            .config("spark.sql.warehouse.dir", f"{hadoop_fs}/user/{username}/final/spark/warehouse")
            .config("spark.driver.memory", "8g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.executor.memory", "6g")
            .config("spark.executor.cores", "4")
            .config("spark.executor.instances", "4")
            .config("spark.dynamicAllocation.enabled", "false")
        )

    spark = builder.master(master).getOrCreate()

    return spark

