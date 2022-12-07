import os
import sys
from os import path

from setuptools import setup, find_packages
import versioneer


min_version = (3, 9)
if sys.version_info < min_version:
    error = """
aimmdb does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have the latest version.

Upgrade pip like so:

pip install --upgrade pip
""".format(
        *(sys.version_info[:2] + min_version)
    )
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

share_aimmdb = os.path.join(here, "share", "aimmdb")


def get_data_files():
    """Get data files in share/aimmdb"""

    data_files = []
    for (d, _dirs, filenames) in os.walk(share_aimmdb):
        rel_d = os.path.relpath(d, here)
        data_files.append((rel_d, [os.path.join(rel_d, f) for f in filenames]))
    return data_files


setup(
    name="aimmdb",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=["docs", "aimmdb/_tests"]),
    install_requires=[
        "tiled[all]",
        "psycopg2-binary",
        "pymongo",
        "pyarrow",
        "h5py",
    ],
    python_requires=">={}".format(".".join(str(n) for n in min_version)),
    extras_require={"test": ["mongomock", "pytest"]},
    entry_points={
        "tiled.structure_client": [
            "MongoAdapter = aimmdb.client:MongoCatalog",
            "AIMMCatalog = aimmdb.client:AIMMCatalog",
            "XAS = aimmdb.client:XASClient",
            "XAS_TEY = aimmdb.client:XASClient",
            "XAS_TFY = aimmdb.client:XASClient",
            "XAS_trans = aimmdb.client:XASClient",
        ],
    },
    data_files=get_data_files(),
    package_data={"aimmdb": ["data/*"]},
)
