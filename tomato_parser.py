import polars as pl
from polars import col as c

vehicle_classes = ['HT', 'MT', 'LT', 'TD', 'SPG']
nations = ['China', 'Czech', 'France', 'Germany', 'Italy', 'Japan', 'Poland', 'Sweden', 'UK', 'USA', 'USSR']


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
    """
    Downloads tomato dataset from kaggle using Kaggle API.
    requires token in ``~/.kaggle/kaggle.json``
    :param path: file path to save dataset
    :return:
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(r'goldflag/10-million-world-of-tanks-battles-from-tomato-gg', 'tomato.csv', path)


def scan_with_validation(tomato_file: str) -> pl.LazyFrame:
    """
    Reads tomato dataset and filters out invalid battle records, attaches deduced ``battle_id`` to records.
    :param tomato_file: path to raw tomato dataset file
    :return: LazyFrame
    """
    return (
        pl.scan_csv(tomato_file)
        .filter(pl.len().over(half_id) % 30 == 0)
        .select((pl.arange(0, pl.len()).alias('battle_id') // 30).set_sorted(), pl.all())
        .filter(filters)
    )


def p_row_to_battle_full(data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """
    Aggregates player records into battle records with all tank ids.
    :param data: Data/LazyFrame with player records from VALID battles.
    :return: LazyFrame
    """
    return (
        data.lazy()
        .sort(['battle_id', 'spawn', 'tier', c('class').replace_strict({'HT': 0, 'MT': 1, 'TD': 2, 'LT': 3, 'SPG': 4})],
              descending=[False, False, True, False])
        .group_by('battle_id', maintain_order=True)
        .agg(
            pl.when(c('won').filter(c('spawn') == 1).first() == True).then(1)
            .when(c('won').filter(c('spawn') == 2).first() == True).then(2)
            .otherwise(0).alias('winner'),
            c('duration').first(),
            c('display_name').first(),
            c('tank_id').filter(c('spawn') == 1).alias('t1'),
            c('tank_id').filter(c('spawn') == 2).alias('t2'),
        )
        .select(pl.exclude('t1', 't2'), *(c('t1').list.get(i).alias(f't1_{i + 1}') for i in range(15)),
                *(c('t2').list.get(i).alias(f't2_{i + 1}') for i in range(15)))
    )


def p_row_to_battle_agg_class(data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """
    Aggregates player records into battle records with class-wise aggregated vehicles.
    :param data: Data/LazyFrame with player records from VALID battles.
    :return: LazyFrame
    """
    return (
        data.lazy()
        .group_by('battle_id')
        .agg(
            pl.when(c('won').filter(c('spawn') == 1).first() == True).then(1)
            .when(c('won').filter(c('spawn') == 2).first() == True).then(2)
            .otherwise(0).alias('winner'),
            c('duration').first(),
            c('display_name').first(),
            c('tier').min().alias('tier_min'),
            c('tier').max().alias('tier_max'),
            *((c('spawn') == team & c('class') == clazz).sum().alias(f't{team}_{clazz}')
              for team in (1, 2)
              for clazz in vehicle_classes)
        )
    )


def p_row_to_battle_agg_class_tier(data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """
    Aggregates player records into battle records with class and tier wise aggregated vehicles.
    :param data: Data/LazyFrame with player records from VALID battles.
    :return: LazyFrame
    """
    return (
        data.lazy()
        .group_by('battle_id')
        .agg(
            pl.when(c('won').filter(c('spawn') == 1).first() == True).then(1)
            .when(c('won').filter(c('spawn') == 2).first() == True).then(2)
            .otherwise(0).alias('winner'),
            c('duration').first(),
            c('display_name').first(),
            *((c('spawn') == team & c('tier') == tier & c('class') == clazz).sum()
            .alias(f't{team}_{tier}_{clazz}')
              for team in (1, 2)
              for tier in range(8, 12)
              for clazz in vehicle_classes)
        )
    )


if __name__ == '__main__':
    smoll_sample = pl.scan_csv('data/parsed_tomato.csv').head(90).collect()

    with pl.Config(tbl_cols=-1, tbl_width_chars=500):
        print(p_row_to_battle_full(smoll_sample).collect())
        print(p_row_to_battle_agg_class(smoll_sample).collect())
        print(p_row_to_battle_agg_class_tier(smoll_sample).collect())
