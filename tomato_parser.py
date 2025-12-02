import polars as pl
from polars import col as c


def stat(name, team):
    return pl.col(name).filter(pl.col('spawn') == team).sum().over('battle_id')


half_id = (
    (
            pl.col('display_name').ne(pl.col('display_name').shift())
            | pl.col('duration').ne(pl.col('duration').shift())
            | pl.col('battle_time').ne(pl.col('battle_time').shift())
    )
    .fill_null(True)
    .cum_sum()
    .set_sorted()
)

filters = (
    c('won').sum().over('battle_id').is_in((15, 0)),
    stat('won', 1).is_in((15, 0)),
    stat('won', 2).is_in((15, 0)),
    stat('tier', 1) == stat('tier', 2),
    stat('damage', 1) <= stat('max_health', 2),
    stat('damage', 2) <= stat('max_health', 1),
    stat('spotting_assist', 1) <= stat('damage', 1),
    stat('spotting_assist', 2) <= stat('damage', 2),
    stat('tracking_assist', 1) <= stat('damage', 1),
    stat('tracking_assist', 2) <= stat('damage', 2),
    stat('penetrations', 1) == stat('penetrations_received', 2),
    stat('penetrations', 2) == stat('penetrations_received', 1),
)


def download_tomato_dataset(path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(r'goldflag/10-million-world-of-tanks-battles-from-tomato-gg', 'tomato.csv', path)


def scan_with_validation(tomato_file: str) -> pl.LazyFrame:
    return (
        pl.scan_csv(tomato_file)
        .filter(pl.len().over(half_id) % 30 == 0)
        .select((pl.arange(0, pl.len()).alias('battle_id') // 30).set_sorted(), pl.all())
        .filter(filters)
    )
