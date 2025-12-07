import polars as pl
from polars import col as c
import tomato_parser as tp

lf = pl.scan_csv('./data/parsed_tomato.csv')

queries = {
    'avg_nation_count': (
        lf
        .group_by('battle_id')
        .agg(*((pl.col('nation') == n).sum().alias(n) for n in tp.nations))
        .select(pl.exclude('battle_id').mean())
    ),
    'avg_nation_count_per_tier': (
        lf
        .group_by('battle_id')
        .agg(c('tier').mean(), *((pl.col('nation') == n).sum().alias(n) for n in tp.nations))
        .group_by('tier')
        .agg(pl.exclude('battle_id').mean())
        .sort('tier')
        # .with_columns(pl.sum_horizontal(pl.exclude('tier')).alias('check'))
    ),
    'avg_type_count': (
        lf
        .group_by('battle_id')
        .agg(*((pl.col('class') == c).sum().alias(c) for c in tp.vehicle_classes))
        .select(pl.exclude('battle_id').mean())
    ),
    'avg_type_count_per_tier': (
        lf
        .group_by('battle_id')
        .agg(c('tier').mean(), *((pl.col('class') == c).sum().alias(c) for c in tp.vehicle_classes))
        .group_by('tier')
        .agg(pl.exclude('battle_id').mean())
        .sort('tier')
        # .with_columns(pl.sum_horizontal(pl.exclude('tier')).alias('check'))
    ),
    'avg_time_per_map': (
        lf
        .group_by('battle_id')
        .agg(c('display_name').first(), c('duration').first())
        .group_by('display_name')
        .agg(c('duration').mean())
        .sort('duration')
    ),
    'avg_time_per_tier': (
        lf
        .group_by('battle_id')
        .agg(c('tier').mean(), c('duration').first())
        .group_by('tier')
        .agg(c('duration').mean())
        .sort('tier')
    ),
    'Side winrate': (
        lf
        .group_by('battle_id')
        .agg(
            c('display_name').first(),
            c('won').filter(c('spawn') == 1).first().alias('Team 1'),
            c('won').filter(c('spawn') == 2).first().alias('Team 2'),
        )
        .group_by('display_name')
        .agg(c('Team 1').mean(), c('Team 2').mean())
        .with_columns(
            (c('Team 1') - c('Team 2')).abs().alias('Winrate Diff'),
            (pl.lit(1) - c('Team 1') - c('Team 2')).alias('Draws'),
        )
        .sort('Winrate Diff', descending=True)
    ),
}

with pl.Config(tbl_cols=-1, tbl_rows=50, tbl_width_chars=500):
    data = pl.collect_all(queries.values())
    for name, df in zip(queries, data):
        print(name)
        print(df)
