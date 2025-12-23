import numpy as np
import h5py

from p2igan_bench.data.dataloader import P2IDataModule


def _write_dummy_sequence(path, frames, height=8, width=8):
    data = np.random.randint(0, 255, size=(frames, height, width, 1), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=data)


def _build_cfg(train_root, test_root, valid_root):
    base_mask = {
        "type": "sti",
        "keep": 2,
        "block_sizes": [2],
        "interval": [1, 2],
    }

    return {
        "data": {
            "train": {
                "name": "dummy",
                "data_root": str(train_root),
                "w": 4,
                "h": 4,
                "sample_length": 4,
                "mask": base_mask,
            },
            "test": {
                "data_root": str(test_root),
                "shuffle": False,
            },
            "valid": {
                "data_root": str(valid_root),
                "w": 4,
                "h": 4,
                "mask": base_mask,
                "shuffle": False,
            },
        },
        "train": {
            "batch_size": 1,
            "num_workers": 0,
        },
    }


def test_datamodule_split_configs_share_shape_and_length(tmp_path):
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    valid_root = tmp_path / "valid"
    train_root.mkdir()
    test_root.mkdir()
    valid_root.mkdir()

    _write_dummy_sequence(train_root / "sample0.h5", frames=6)
    _write_dummy_sequence(test_root / "sample0.h5", frames=5)
    _write_dummy_sequence(valid_root / "sample0.h5", frames=7)

    cfg = _build_cfg(train_root, test_root, valid_root)
    data_module = P2IDataModule(cfg)

    train_dataset = data_module.train_dataloader().dataset
    test_dataset = data_module.test_dataloader().dataset
    valid_dataset = data_module.val_dataloader().dataset

    train_video, _, _ = train_dataset[0]
    test_video, _, _ = test_dataset[0]
    valid_video, _, _ = valid_dataset[0]

    assert train_video.shape[0] == 4  # respect train sample_length
    assert test_video.shape[0] == 4  # inherits sample_length from train
    assert valid_video.shape[0] == 7  # no truncation when sample_length omitted

    assert train_video.shape[1:] == (4, 4, 1)
    assert test_video.shape == train_video.shape
    assert valid_video.shape[1:] == (4, 4, 1)
