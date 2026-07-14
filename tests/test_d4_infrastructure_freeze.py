from __future__ import annotations

import hashlib
import json
from pathlib import Path


KNN_ROOT = Path("configs/solidified/knn/Dataset4")
TARGETS = ("166_258", "166_432", "166_433", "166_313", "166_311")

EXPECTED = {
    "with": {
        "file_sha256": "152941091f55d8d3efefb2023b1664d3712ab4cd405cb6ed1527516824183913",
        "candidate_digests": {
            "166_258": "f4ded7bb774ffc7a98ade6e1af7be704942c1960696c2f861c1d2fd05437ac37",
            "166_432": "b6e263579b56eee52c9a1a59910a6a05e45cf761dc3e1511ce0b47f64ab5352c",
            "166_433": "babd34c7e4ef24ddb2856d5b4a7fbc96a3c8f5e1fea69378a65aea6408a83111",
            "166_313": "43a74ecd6fec694c92406b2b84e7ac0745b44f0c39e9034ff7e38009ee67e93c",
            "166_311": "f526c06975c014d3e9edd22a0ff6d28f3a911e8862907dc427f442209705caec",
        },
        "selection_digests": {
            "166_258": "27d83200ada832718b556517b4ec9a97b38996781705d650687f59b62ae3a8a6",
            "166_432": "2515a8b57c15b6c0d40dddf3462fd0f4227b2dce63631cd4badfceb6522881e7",
            "166_433": "b27d93912a85cc23ad09b2d28a7debff19b4133eebcb8902091e7b7dd90341bc",
            "166_313": "02abd64565ed5a784678f85b27b8454dc1e04d24eff9e2b2d49b2c23006a8c4e",
            "166_311": "44f874010105309d95032bdd8ebe5c1e6a03650ffe6fd6f2118b8e8d6bf3c7b5",
        },
        "top3": {
            "166_258": ("729_424", "530_155", "356_242"),
            "166_432": ("620_261", "462_424", "111_261"),
            "166_433": ("675_185", "594_432", "723_432"),
            "166_313": ("528_432", "449_432", "1035_432"),
            "166_311": ("1047_261", "329_261", "447_246"),
        },
    },
    "without": {
        "file_sha256": "289748390dc210a5fec97a45e2c6ec9821f2be4132ab8c16d041dd574383d0ce",
        "candidate_digests": {
            "166_258": "7940a1c51eee067b1bcd5868aee2cf70c53be1aaa9e54dec49c826873658206a",
            "166_432": "286d273987e7ff2ac98e0e53b44539c5a8a2901d786ac5afceb1d024f6abee91",
            "166_433": "6be3acdca16cea9ad34356ed69866a10d2fa9810b3585fe0fb405c069e12b559",
            "166_313": "762c2bda76dec4510af014ce0475a862a5f4748e9df95db5225660f387e359bb",
            "166_311": "db984d919608e4e6a7d149c7248d35f4215d7774d2615b3654134855f1edfdfc",
        },
        "selection_digests": {
            "166_258": "426aedd61c0748c02bed086b862c25a3df8a54101b5ae1171c393350f73664c4",
            "166_432": "a642de6ede6a27949ab7293ad74702af7fe7e94584898b293442827fbd63b616",
            "166_433": "c1a6f28286b81bd21c4f2cdd6e246c234cffac020fe6d103c8062cf67537dde3",
            "166_313": "bc52c3509cad07c69c2afb03b3ec66f662ae5211e8904e88ae01fb4328c023d8",
            "166_311": "3bf294c1df9296cd4c1786cfac18f2580b7241946f84ada8ac07fa8e185aa68b",
        },
        "top3": {
            "166_258": ("166_184", "166_560", "166_530"),
            "166_432": ("166_506", "166_242", "166_510"),
            "166_433": ("166_184", "166_530", "166_560"),
            "166_313": ("166_506", "166_548", "166_510"),
            "166_311": ("166_506", "166_548", "166_510"),
        },
    },
}


def test_checked_d4_knn_artifacts_are_frozen() -> None:
    for mode, expected in EXPECTED.items():
        path = KNN_ROOT / f"knn_{mode}_info_sharing.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["file_sha256"]
        assert payload["info_sharing"] == mode
        assert payload["k"] == 3
        assert tuple(payload["results"]) == TARGETS

        metadata = payload["selection_metadata"]
        assert {
            target: metadata[target]["candidate_pool_digest"] for target in TARGETS
        } == expected["candidate_digests"]
        assert {
            target: metadata[target]["selection_result_digest"] for target in TARGETS
        } == expected["selection_digests"]
        assert {
            target: tuple(row["source_entity"] for row in payload["results"][target])
            for target in TARGETS
        } == expected["top3"]
