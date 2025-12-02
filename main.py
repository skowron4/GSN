import pathlib

from tomato_parser import download_tomato_dataset, scan_with_validation, p_row_to_battle_full, p_row_to_battle_agg_class


def main():
    data_dir = r'./data'
    raw_dataset_path = f'{data_dir}/tomato.csv'
    parsed_dataset_path = f'{data_dir}/parsed_tomato.csv'

    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(raw_dataset_path).is_file():
        download_tomato_dataset(data_dir)
    if not (p := pathlib.Path(parsed_dataset_path)).is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        scan_with_validation(raw_dataset_path).sink_csv(parsed_dataset_path)


if __name__ == '__main__':
    main()
