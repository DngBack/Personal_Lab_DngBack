"""Download Hugging Face datasets for offline analysis."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

ExportFormat = Literal["jsonl", "parquet", "csv"]

_MODULE_DIR = Path(__file__).resolve().parent
_AIR_DATA_ROOT = _MODULE_DIR.parents[1]
_DEFAULT_OUTPUT_ROOT = _AIR_DATA_ROOT / "data" / "hf"
_DEFAULT_REPO_LIST = _AIR_DATA_ROOT / "data" / "data_hf.txt"
_DATA_EXTENSIONS = {".jsonl", ".json", ".parquet", ".csv", ".tsv", ".txt"}


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _resolve_output_dir(repo_id: str, output_dir: Path | str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return _DEFAULT_OUTPUT_ROOT / _sanitize_repo_id(repo_id)


def _save_dataset(
    dataset: Dataset,
    output_path: Path,
    export_format: ExportFormat,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif export_format == "parquet":
        dataset.to_parquet(str(output_path))
    elif export_format == "csv":
        pd.DataFrame(dataset).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    return output_path


def _iter_splits(dataset_obj: Dataset | DatasetDict) -> list[tuple[str, Dataset]]:
    if isinstance(dataset_obj, DatasetDict):
        return [(split_name, split_data) for split_name, split_data in dataset_obj.items()]
    return [("data", dataset_obj)]


def _write_dataset_info(
    info_path: Path,
    *,
    repo_id: str,
    config_name: str | None,
    split: str,
    num_rows: int | None,
    columns: list[str] | None,
    export_format: ExportFormat | str,
    output_path: Path,
    source: str,
) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_id": repo_id,
        "config_name": config_name,
        "split": split,
        "num_rows": num_rows,
        "columns": columns,
        "export_format": export_format,
        "output_path": str(output_path),
        "source": source,
    }

    existing: list[dict[str, Any]] = []
    if info_path.exists():
        existing = json.loads(info_path.read_text(encoding="utf-8"))

    existing = [item for item in existing if item.get("output_path") != str(output_path)]
    existing.append(payload)
    info_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _load_dataset_via_api(
    repo_id: str,
    *,
    config_name: str | None,
    split: str | None,
    cache_dir: str | None,
) -> Dataset | DatasetDict:
    load_kwargs: dict[str, Any] = {"path": repo_id, "cache_dir": cache_dir}
    if config_name is not None:
        load_kwargs["name"] = config_name
    if split is not None:
        return load_dataset(**load_kwargs, split=split)
    return load_dataset(**load_kwargs)


def _download_via_datasets_api(
    repo_id: str,
    *,
    config_name: str | None,
    split: str | None,
    base_output_dir: Path,
    export_format: ExportFormat,
    cache_dir: str | None,
    download_all_configs: bool,
) -> dict[str, Path]:
    exported: dict[str, Path] = {}

    config_names: list[str | None]
    if download_all_configs:
        config_names = get_dataset_config_names(repo_id, cache_dir=cache_dir)
    elif config_name is not None:
        config_names = [config_name]
    else:
        config_names = [None]

    for cfg in tqdm(config_names, desc=f"Loading {repo_id} via datasets"):
        dataset_obj = _load_dataset_via_api(
            repo_id,
            config_name=cfg,
            split=split,
            cache_dir=cache_dir,
        )

        cfg_label = cfg or "default"
        cfg_output_dir = base_output_dir / cfg_label

        for split_name, split_data in _iter_splits(dataset_obj):
            ext = {"jsonl": "jsonl", "parquet": "parquet", "csv": "csv"}[export_format]
            output_path = cfg_output_dir / f"{split_name}.{ext}"
            _save_dataset(split_data, output_path, export_format)

            key = f"{cfg_label}/{split_name}.{ext}"
            exported[key] = output_path

            _write_dataset_info(
                base_output_dir / "dataset_info.json",
                repo_id=repo_id,
                config_name=cfg,
                split=split_name,
                num_rows=len(split_data),
                columns=list(split_data.column_names),
                export_format=export_format,
                output_path=output_path,
                source="datasets",
            )

    return exported


def _list_hub_data_files(repo_id: str, config_name: str | None) -> list[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    data_files = [
        remote_path
        for remote_path in files
        if Path(remote_path).suffix.lower() in _DATA_EXTENSIONS
        and not remote_path.startswith(".")
    ]

    if config_name is None:
        return sorted(data_files)

    matched = [
        remote_path
        for remote_path in data_files
        if config_name in Path(remote_path).stem or f"/{config_name}." in remote_path
    ]
    return sorted(matched)


def _download_via_hub_files(
    repo_id: str,
    *,
    config_name: str | None,
    base_output_dir: Path,
    cache_dir: str | None,
) -> dict[str, Path]:
    data_files = _list_hub_data_files(repo_id, config_name)
    if not data_files:
        hint = f" (config={config_name!r})" if config_name else ""
        raise FileNotFoundError(f"No data files found in repo {repo_id}{hint}")

    exported: dict[str, Path] = {}
    for remote_path in tqdm(data_files, desc=f"Downloading files from {repo_id}"):
        cached_path = Path(
            hf_hub_download(
                repo_id,
                remote_path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        )
        output_path = base_output_dir / remote_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, output_path)

        exported[remote_path] = output_path

        num_rows = None
        columns = None
        export_format: ExportFormat | str = output_path.suffix.lstrip(".")
        if output_path.suffix == ".jsonl":
            num_rows = _count_jsonl_rows(output_path)
            with output_path.open("r", encoding="utf-8") as f:
                columns = list(json.loads(f.readline()).keys())

        _write_dataset_info(
            base_output_dir / "dataset_info.json",
            repo_id=repo_id,
            config_name=config_name or Path(remote_path).stem,
            split=Path(remote_path).stem,
            num_rows=num_rows,
            columns=columns,
            export_format=export_format,
            output_path=output_path,
            source="hub_files",
        )

    return exported


def download_hf_dataset(
    repo_id: str,
    *,
    config_name: str | None = None,
    split: str | None = None,
    output_dir: Path | str | None = None,
    export_format: ExportFormat = "jsonl",
    cache_dir: str | None = None,
    download_all_configs: bool = False,
    prefer_hub_files: bool = False,
) -> dict[str, Path]:
    """Download a Hugging Face dataset and export it for local analysis.

    Uses the ``datasets`` API when possible. For legacy repos that still ship
    a loading script (e.g. ``L4NLP/LEval``), falls back to downloading raw
    data files from the Hub.

    Args:
        repo_id: Dataset repo id, e.g. ``"L4NLP/LEval"``.
        config_name: Optional subset/config name, e.g. ``"gsm100"``.
        split: Optional split name, e.g. ``"test"``. If omitted, all splits are saved.
        output_dir: Directory to write exported files. Defaults to ``air_data/data/hf/<repo_id>``.
        export_format: One of ``"jsonl"``, ``"parquet"``, or ``"csv"``.
        cache_dir: Optional Hugging Face cache directory.
        download_all_configs: If True, download every config/subset in the dataset repo.
        prefer_hub_files: Skip ``datasets`` and download raw files from the Hub.

    Returns:
        Mapping of exported file keys to output paths.
    """
    base_output_dir = _resolve_output_dir(repo_id, output_dir)

    if prefer_hub_files:
        return _download_via_hub_files(
            repo_id,
            config_name=None if download_all_configs else config_name,
            base_output_dir=base_output_dir,
            cache_dir=cache_dir,
        )

    try:
        return _download_via_datasets_api(
            repo_id,
            config_name=config_name,
            split=split,
            base_output_dir=base_output_dir,
            export_format=export_format,
            cache_dir=cache_dir,
            download_all_configs=download_all_configs,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

    return _download_via_hub_files(
        repo_id,
        config_name=None if download_all_configs else config_name,
        base_output_dir=base_output_dir,
        cache_dir=cache_dir,
    )


def read_repo_list(path: Path | str | None = None) -> list[str]:
    """Read repo ids from a text file (one repo id per line, ``#`` comments allowed)."""
    repo_list_path = Path(path or _DEFAULT_REPO_LIST).expanduser().resolve()
    if not repo_list_path.exists():
        raise FileNotFoundError(f"Repo list not found: {repo_list_path}")

    repos: list[str] = []
    for line in repo_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def download_from_repo_list(
    path: Path | str | None = None,
    *,
    export_format: ExportFormat = "jsonl",
    cache_dir: str | None = None,
    download_all_configs: bool = False,
    prefer_hub_files: bool = False,
) -> dict[str, dict[str, Path]]:
    """Download every repo id listed in ``data_hf.txt``."""
    results: dict[str, dict[str, Path]] = {}
    for repo_id in read_repo_list(path):
        results[repo_id] = download_hf_dataset(
            repo_id,
            export_format=export_format,
            cache_dir=cache_dir,
            download_all_configs=download_all_configs,
            prefer_hub_files=prefer_hub_files,
        )
    return results


def preview_dataset(
    repo_id: str,
    config_name: str | None = None,
    split: str | None = None,
    cache_dir: str | None = None,
) -> None:
    """Print a quick summary and one sample row."""
    try:
        dataset_obj = _load_dataset_via_api(
            repo_id,
            config_name=config_name,
            split=split,
            cache_dir=cache_dir,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        data_files = _list_hub_data_files(repo_id, config_name)
        if not data_files:
            raise

        first_file = data_files[0]
        cached_path = Path(
            hf_hub_download(repo_id, first_file, repo_type="dataset", cache_dir=cache_dir)
        )
        print(f"Repo: {repo_id}")
        print(f"Fallback: raw hub file preview ({first_file})")
        if cached_path.suffix == ".jsonl":
            with cached_path.open("r", encoding="utf-8") as f:
                sample = json.loads(f.readline())
                row_count = 1 + sum(1 for _ in f)
            print(f"Rows in {first_file}: {row_count}")
            print(f"Columns: {list(sample.keys())}")
            print("Sample row:")
            print(json.dumps(sample, ensure_ascii=False, indent=2)[:2000])
            return

        print(f"Downloaded sample file: {cached_path}")
        return

    if isinstance(dataset_obj, DatasetDict):
        print(f"Repo: {repo_id}")
        print(f"Configs loaded: {list(dataset_obj.keys()) if config_name is None else config_name}")
        first_split = next(iter(dataset_obj))
        sample_ds = dataset_obj[first_split]
        print(f"Preview split: {first_split} ({len(sample_ds)} rows)")
        print(f"Columns: {sample_ds.column_names}")
        print("Sample row:")
        print(json.dumps(sample_ds[0], ensure_ascii=False, indent=2)[:2000])
        return

    print(f"Repo: {repo_id}")
    print(f"Rows: {len(dataset_obj)}")
    print(f"Columns: {dataset_obj.column_names}")
    print("Sample row:")
    print(json.dumps(dataset_obj[0], ensure_ascii=False, indent=2)[:2000])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Hugging Face datasets for analysis.")
    parser.add_argument("repo_id", nargs="?", default="L4NLP/LEval", help="Dataset repo id")
    parser.add_argument("--config", dest="config_name", default=None, help="Dataset config/subset name")
    parser.add_argument("--split", default=None, help="Dataset split, e.g. test")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: air_data/data/hf/<repo_id>)",
    )
    parser.add_argument(
        "--format",
        dest="export_format",
        choices=("jsonl", "parquet", "csv"),
        default="jsonl",
        help="Export format when using the datasets API",
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache dir")
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Download all configs/subsets of the dataset",
    )
    parser.add_argument(
        "--hub-files",
        action="store_true",
        help="Download raw data files from the Hub instead of using datasets.load_dataset",
    )
    parser.add_argument(
        "--from-file",
        action="store_true",
        help="Download all repo ids listed in air_data/data/data_hf.txt",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only print dataset summary/sample, do not export files",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.preview:
        preview_dataset(
            args.repo_id,
            config_name=args.config_name,
            split=args.split,
            cache_dir=args.cache_dir,
        )
        return

    if args.from_file:
        results = download_from_repo_list(
            export_format=args.export_format,
            cache_dir=args.cache_dir,
            download_all_configs=args.all_configs,
            prefer_hub_files=args.hub_files,
        )
        for repo_id, exported in results.items():
            print(f"\n{repo_id}:")
            for key, path in exported.items():
                print(f"  {key} -> {path}")
        return

    exported = download_hf_dataset(
        args.repo_id,
        config_name=args.config_name,
        split=args.split,
        output_dir=args.output_dir,
        export_format=args.export_format,
        cache_dir=args.cache_dir,
        download_all_configs=args.all_configs,
        prefer_hub_files=args.hub_files,
    )

    print(f"Downloaded {args.repo_id}")
    for key, path in exported.items():
        print(f"  {key} -> {path}")


if __name__ == "__main__":
    main()
