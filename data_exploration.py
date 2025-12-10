import polars as pl
from polars import col as c
import tomato_parser as tp
import seaborn as sns
import matplotlib.pyplot as plt

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

for (name, df) in zip(queries.keys(), data):
    pdf = df.to_pandas()
    print(f"Plotting: {name}")

    plt.figure(figsize=(10, 6))

    # ====== AVG NATION COUNT ======
    if name == "avg_nation_count":
        pdf_sorted = pdf.sort_values(pdf.columns[0], ascending=False)
        sns.barplot(x=pdf_sorted.columns, y=pdf_sorted.iloc[0])
        plt.ylabel("Average per battle")
        plt.title("Average Nation Count")

    # ====== AVG NATION COUNT PER TIER ======
    elif name == "avg_nation_count_per_tier":
        pdf = pdf.melt(id_vars=["tier"], var_name="nation", value_name="count")
        sns.lineplot(data=pdf, x="tier", y="count", hue="nation", marker="o")
        plt.title("Average Nation Count per Tier")

    # ====== AVG TYPE COUNT ======
    elif name == "avg_type_count":
        row = pdf.iloc[0]
        sorted_row = row.sort_values(ascending=False)
        sns.barplot(x=sorted_row.index, y=sorted_row.values)
        plt.ylabel("Average per battle")
        plt.title("Average Vehicle Type Count")

    # ====== AVG TYPE COUNT PER TIER ======
    elif name == "avg_type_count_per_tier":
        pdf = pdf.melt(id_vars=["tier"], var_name="class", value_name="count")
        sns.lineplot(data=pdf, x="tier", y="count", hue="class", marker="o")
        plt.title("Average Vehicle Type Count per Tier")

    # ====== AVG TIME PER MAP ======
    elif name == "avg_time_per_map":
        pdf_sorted = pdf.sort_values("duration")
        sns.barplot(data=pdf_sorted, x="duration", y="display_name", orient="h")
        plt.xlabel("Average duration (s)")
        plt.ylabel("Map")
        plt.title("Average Battle Duration per Map")

    # ====== AVG TIME PER TIER ======
    elif name == "avg_time_per_tier":
        sns.lineplot(data=pdf, x="tier", y="duration", marker="o")
        plt.ylabel("Average duration (s)")
        plt.title("Average Battle Duration per Tier")

    # ====== SIDE WINRATE ======
    elif name == "Side winrate":
        melted = pdf.melt(id_vars=["display_name"], value_vars=["Team 1", "Team 2"], var_name="team", value_name="winrate")
        melted["winrate"] *= 100
        pdf["Winrate Diff %"] = pdf["Winrate Diff"] * 100
        pdf["Draws %"] = pdf["Draws"] * 100

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.barplot(data=melted, x="winrate", y="display_name", hue="team", orient="h", ax=axes[0])
        axes[0].set_title("Team Winrate per Map (%)")

        diff_sorted = pdf.sort_values("Winrate Diff %", ascending=False)
        sns.barplot(data=diff_sorted, x="Winrate Diff %", y="display_name", orient="h", ax=axes[1])
        axes[1].set_title("Winrate Difference per Map (%)")

        plt.tight_layout()
        plt.show()
        continue  # skip default show below

    plt.tight_layout()
    plt.show()