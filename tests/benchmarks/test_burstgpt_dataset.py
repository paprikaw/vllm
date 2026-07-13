import pandas as pd

from benchmarks.benchmark_dataset import BurstGPTDataset


def _write_dataset(tmp_path):
    path = tmp_path / "burstgpt.csv"
    pd.DataFrame([
        [0.0, "ChatGPT", 10, 2, 12],
        [1.0, "GPT-4", 20, 3, 23],
        [2.0, "ChatGPT", 30, 0, 30],
    ], columns=[
        "Timestamp", "Model", "Request tokens", "Response tokens",
        "Total tokens"
    ]).to_csv(path, index=False)
    return path


def test_burstgpt_dataset_all_model_filter(tmp_path):
    dataset = BurstGPTDataset(
        dataset_path=str(_write_dataset(tmp_path)), model_filter="ALL")

    assert dataset.data["Model"].tolist() == ["ChatGPT", "GPT-4"]


def test_burstgpt_dataset_specific_model_filter(tmp_path):
    dataset = BurstGPTDataset(
        dataset_path=str(_write_dataset(tmp_path)), model_filter="ChatGPT")

    assert dataset.data["Model"].tolist() == ["ChatGPT"]
    assert dataset.data["Response tokens"].tolist() == [2]
