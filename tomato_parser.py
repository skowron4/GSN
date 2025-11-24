import polars as pl
from polars import col as c

half_id = (
    (
            pl.col('display_name').ne(pl.col('display_name').shift())
            | pl.col('duration').ne(pl.col('duration').shift())
            | pl.col('battle_time').ne(pl.col('battle_time').shift())
    )
    .fill_null(True)
    .cum_sum()
)

filters = (
    pl.len().over('battle_id', 'spawn') == 15,
    c.won.min().over('battle_id', 'spawn') == c.won.max().over('battle_id', 'spawn'),
    c.won.sum().over('battle_id').is_in((0, 15)),
    c.tier.filter(c.spawn == 1).sum().over('battle_id') == c.tier.filter(c.spawn == 2).sum().over('battle_id'),
    c.damage.filter(c.spawn == 1).sum().over('battle_id') <= c.max_health.filter(c.spawn == 2).sum().over('battle_id'),
    c.damage.filter(c.spawn == 2).sum().over('battle_id') <= c.max_health.filter(c.spawn == 1).sum().over('battle_id'),
    c.spotting_assist.sum().over('battle_id', 'spawn') <= c.damage.sum().over('battle_id', 'spawn'),
    c.tracking_assist.sum().over('battle_id', 'spawn') <= c.damage.sum().over('battle_id', 'spawn'),
    c.penetrations.filter(c.spawn == 1).sum().over('battle_id')
    == c.penetrations_received.filter(c.spawn == 2).sum().over('battle_id'),
    c.penetrations.filter(c.spawn == 2).sum().over('battle_id')
    == c.penetrations_received.filter(c.spawn == 1).sum().over('battle_id'),
)

(
    pl.scan_csv('tomato.csv')
    .filter(pl.len().over(half_id) % 30 == 0)
    .select(pl.arange(0, pl.len()).alias('battle_id') // 30, pl.all())
    .set_sorted('battle_id')
    .filter(filters)
    .sink_csv("parsed_tomato.csv")
)
