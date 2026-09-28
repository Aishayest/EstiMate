"""Download the Deep-SE story point dataset (Choetkiertikul et al., 2018) into data/raw/.

Source: https://github.com/morakotch/datasets/tree/master/storypoint/IEEE%20TSE2018/dataset
Each file is verified against a SHA-256 checksum so results are reproducible.

Usage: python -m src.download_data
"""
import hashlib
import ssl
import urllib.request

import certifi

from src import config

BASE_URL = (
    "https://raw.githubusercontent.com/morakotch/datasets/master/"
    "storypoint/IEEE%20TSE2018/dataset/"
)

SHA256 = {
    "appceleratorstudio": "43f2de728c4251084782cc13bcc339e67e4cbbfc1d6f07be9d2be6b799bbf3b4",
    "aptanastudio": "c7af639b52a5a679dbfb29ac65d5c8558c18af45946932cd4e89a73c2ff34172",
    "bamboo": "90d64e163896168edad7ad5ed4411fcd0da103cf6d9c0bcde67fa97add2f1db9",
    "clover": "8218c05d864d2130e88ae24b6a90f318c5bba3e76692b132dbf51880f6c73736",
    "datamanagement": "5bed6be21916f86aeba375a06e4d2e3466257f5c2be78201e56effe26f0053a8",
    "duracloud": "ca9c2252eea0848cec3724e864ce782f4f1ac16838ebad19128bf67068f306cf",
    "jirasoftware": "2e6f525c0d88ed36985c287e9cdc42bf6800f66be0e4e900854368e98cd5c6aa",
    "mesos": "192814946b6d61e01a2d068cc957eade8402b40a864d6be42032f73aaa7f891e",
    "moodle": "5aff0df8cf96cdaa4295341c6e35fe20c550f088979ca07cee74bc9b8f3397ce",
    "mule": "a4e7cf46f94ba3ece298c24c49eee41c195326d65a6451f40fcf2f6fbe4759a2",
    "mulestudio": "e2a6019d1bcce88845667fe28d7c94212c181c030462b9805953490f756411cc",
    "springxd": "edbef829fbfb9dae56085db0804ce3077aa3f44ca65f95f80909750d301abbc5",
    "talenddataquality": "43d1cfab94d8fdcb6ddd85131a94957e18d3b0335b194fd4030a18e0c49bfaeb",
    "talendesb": "4929b8627e14c8fee520d35ec34c22f1cc0f407729837a4584ccbaad8029c322",
    "titanium": "a1ecb54d8a8a15b9c93f9fbdc5868404719c1df063504321375fdf98106aa9b5",
    "usergrid": "71fabca08e9172cc401ba57ad4dd138dbb17fbe46fcccd883d1452248aed08a6",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    # certifi's CA bundle avoids SSL failures on python.org macOS builds.
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp:
        return resp.read()


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        path = config.RAW_DIR / f"{name}.csv"
        if path.exists() and sha256(path.read_bytes()) == expected:
            print(f"ok       {name}")
            continue
        data = fetch(BASE_URL + f"{name}.csv")
        if sha256(data) != expected:
            raise RuntimeError(f"Checksum mismatch for {name}.csv")
        path.write_bytes(data)
        print(f"fetched  {name}")


if __name__ == "__main__":
    main()
