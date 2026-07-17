from __future__ import annotations

import hashlib
import json
from pathlib import Path


KNN_ROOT = Path("configs/solidified/knn/Dataset4")
TARGETS = ("166_258", "166_432", "166_433", "166_313", "166_311")

EXPECTED = {
    "with": {
        "file_sha256": "44a332c374f97fce833b284802122f5337f8376a3bc1d70110c0c67debd5db6a",
        "manifest_identity_digest": "231698a4ac844811087b9563f629e981afa50510dd3cf99e847653abdbd59bf4",
        "candidate_digests": {
            "166_258": "8bef9aa02a5298bd177552cf765532d5b44f229d8791b3f1733bf1f50aa6d6da",
            "166_432": "e6e406f3c286e6c02109505a706a8a01bd8de8b2bc9e797c3fd1cd47793524c6",
            "166_433": "4f22963478294fae6231c09bafb61dc196850eea651e8170b74ced572256d9df",
            "166_313": "f5560eb31c03566bc6e19c1cbc3c5fb450d648b4a49660abab50d17f62bc66c1",
            "166_311": "097ad76ac2d1f59d20bf56bb738889c9b7be681b8cc693a167cc761f59ccd96c",
        },
        "selection_digests": {
            "166_258": "4196c419072859d0421fae4ad77e608ecf7d7987602fd882148e8764d444fb74",
            "166_432": "394b098363b7421f74c9b3890b0e572fdf9ef9887cc9ce34d5d594845afd2112",
            "166_433": "8ba95d954ce553e933d747fca8a27aa6e4387b5b1d8124bf1be9a44484d3828c",
            "166_313": "91fce8776ba14d7aa9d27157d08fe670a39ad37675cd2b54b1e872dd471bd7c8",
            "166_311": "8cce11adb2a7c5c7ad95b3448765d51c73d9d04987ae9cd7d3f7ee9b9093ab5f",
        },
        "source_pool_fingerprints": {
            "166_258": "4523fa81be8833bbab079a64480cf219eb87d0ec8a8e928fa5e9f167bdb45b6f",
            "166_432": "d2fb4881382827374056d56936e0ba0669ba86ea89e955ae55708b203e567a54",
            "166_433": "6c795f900a1b14b9742827d0c39eb4565ce0e4289187565759309a7c8865d28c",
            "166_313": "e253ea0043709540fa279c070f7a5818e8f9f35129162565e478537f8b513522",
            "166_311": "a5f80a068f3fd067975655d7f3c113a5ad632d00b169ff353c9edf4f098908be",
        },
        "consumer_fingerprints": {
            "166_258": "abbc355c3b2fb84754dff185faf05a99f7df84ed717b14b93e035c7a7c172416",
            "166_432": "c78593768bcb7a200e40fb4f0c8f0f68f4ad5afbea95301e1504585b99af8144",
            "166_433": "dff407b58c32087961cab693092aa0d7ddf7d2825cb14887de0eb421ff758899",
            "166_313": "ef353b3b56821444d703ddcf0a132f54aea085f627092d56857498373b8530bf",
            "166_311": "61724fd5ae2cac7aabe5976471296993118025d5d5ce2d355f3231dd542d9a00",
        },
        "top3": {
            "166_258": ("729_424", "530_155", "356_242"),
            "166_432": ("620_261", "462_424", "111_261"),
            "166_433": ("675_185", "594_432", "723_432"),
            "166_313": ("64_313", "710_313", "528_432"),
            "166_311": ("1047_261", "329_261", "447_246"),
        },
    },
    "without": {
        "file_sha256": "b61df584146d27c5c4dc154c4d6445ddc298200e888cc186bedb51601091a73e",
        "manifest_identity_digest": "372596182ec34f64b68455f8d70efabe4153cb7d1233a8fef7d56ac17ee61ea4",
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
        "source_pool_fingerprints": {
            "166_258": "421d90f0213967a36c1b6d5cace64ff8c73424ffa68abbf6b740f12be797ec2e",
            "166_432": "827f592cf96574181d73c2340739a24f9204dc183613439450ed52511ff02e65",
            "166_433": "f1d575571cb46cbe6cfa86b36211df5dd06717d367cfc1aca2126e72606b0a19",
            "166_313": "97b73fab3834f4bd32adda1badc8105c77499f137fa6ea2e6ccb35aad7948642",
            "166_311": "41fcb24fc930f487679f8955772c51041ed303552b1f960e6a05a279361e4d9d",
        },
        "consumer_fingerprints": {
            "166_258": "f56294416199219e654353cc80f34d2503b3a771d95a8f963cfd67573f639063",
            "166_432": "77358e913dba5f8e5497184b4ed9df176e943c00099c479a2eac26417268aee4",
            "166_433": "77835e907e7f30cc0e6430ff90e84003bf5c4d7b6e3b0919734f23aa3d24fc35",
            "166_313": "b737526d43c47a8ea97842d320041a0d6613940b80b578e613a66a1350c11167",
            "166_311": "9eef112b00b8013ab01bae76e9ae293d06ff0c80136161476bc4341d6d974d84",
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
        assert (
            payload["d4_manifest_identity"]["manifest_identity_digest"]
            == expected["manifest_identity_digest"]
        )
        assert {
            target: metadata[target]["candidate_pool_digest"] for target in TARGETS
        } == expected["candidate_digests"]
        assert {
            target: metadata[target]["selection_result_digest"] for target in TARGETS
        } == expected["selection_digests"]
        assert {
            target: metadata[target]["source_pool_fingerprint"] for target in TARGETS
        } == expected["source_pool_fingerprints"]
        assert {
            target: metadata[target]["consumer_fingerprint"] for target in TARGETS
        } == expected["consumer_fingerprints"]
        assert {
            target: tuple(row["source_entity"] for row in payload["results"][target])
            for target in TARGETS
        } == expected["top3"]
