import subprocess


def test_cli():
    _ = subprocess.run(
        [
            "matchviz",
            "save-points"
            "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/",
            "points" "--ngjson",
            "test.json",
        ]
    )
