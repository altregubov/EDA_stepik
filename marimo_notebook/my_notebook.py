# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "marimo>=0.21.1",
#     "numpy==2.4.3",
#     "polars==1.39.3",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs
    import altair as alt
    import vegafusion as vf
    alt.data_transformers.enable("vegafusion")
    return alt, cs, pl


@app.cell
def _(fill_medians, pl):
    df = (
        pl.read_csv("https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/diamonds_good.csv")
        .pipe(rename_df)
        .unique(keep="first")
        .pipe(fill_medians, ["carat", "depth", "y"])
        .with_columns(
            pl.col("cut").replace("Goood", "Good")
        )
    ).filter((pl.col("x") > 0) & (pl.col("y") > 0) & (pl.col("z") > 0))
    return (df,)


@app.function
def rename_df(dataframe):
    rename_mapping = {
        "'x'": "x",
        "'y'": "y",
        "'z'": "z",
    }
    return dataframe.rename(rename_mapping)


@app.cell
def _(cs, pl):
    def fill_medians(dataframe: pl.DataFrame, cols: [str]) -> pl.DataFrame:
        # cs.by_name(*cols) выбирает все указанные колонки без циклов Python
        return dataframe.with_columns(
            cs.by_name(cols).fill_null(cs.by_name(cols).median())
        )

    return (fill_medians,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(alt, cs, df):
    # Гистограммы распределения по всем колонкам
    alt.Chart(df, width=200, height=200).mark_bar().encode(
        x=alt.X(alt.repeat('repeat'), type='quantitative', bin=alt.Bin(maxbins=100)),
        y='count()',
    ).repeat(
        repeat=df.select(cs.numeric()).columns,
        columns=3
    )
    return


@app.cell
def _(alt, df):
    # Carat
    alt.Chart(df, width=400, height=400).mark_bar().encode(
        x=alt.X("carat:Q").bin(maxbins=20),
        y='count()',
    )
    return


@app.cell
def _(df, pl):
    df.select(
        pl.col("carat").min().alias("min_carat"),
        pl.col("carat").max().alias("max_carat")
    )
    return


@app.cell
def _(alt, df):
    alt.Chart(df, width=400, height=400).mark_circle().encode(
        x='carat',
        y='price'
    )
    return


@app.cell
def _(df):
    #Work with cut
    df['cut'].value_counts()
    return


@app.cell
def _(alt, df):
    alt.Chart(df, width=400, height=400).mark_bar().encode(
        x='cut',
        y='mean(price)'
    )
    return


@app.cell
def _(alt, df):
    alt.Chart(df, width=400, height=400).mark_bar().encode(
        x='cut',
        y='count(cut)'
    )
    return


@app.cell
def _(df, pl):
    df.group_by('cut').agg(pl.col('price').mean().cast(int))
    return


@app.cell
def _(df, pl):
    df.group_by("cut").agg(
            pl.col("price").count().alias("Count"),
            pl.col("price").mean().alias("PriceMean").cast(int),
            pl.col("price").median().alias("PriceMedian").cast(int),
            pl.col("carat").mean().alias("CaratMean").round(1)
        ).sort("PriceMean", descending=True)
    return


@app.cell
def _(alt, df):
    alt.Chart(df, width=300, height=300).mark_bar().encode(
        x=alt.X("price", type='quantitative', bin=alt.Bin(maxbins=20)),
        y='count()',
        color = alt.Color("cut:N", legend = None)
    ).facet(
        facet='cut:N',
        columns=3
    )
    return


@app.cell
def _(df, pl):
    #single correlation calculation
    df.select(pl.corr("table", "price", method="spearman"))
    return


@app.cell
def _(df, pl):
    #single correlation calculation
    df.select(pl.corr("depth", "price", method="spearman"))
    return


@app.cell
def _(df, pl):
    df.group_by("clarity").agg(
            pl.col("clarity").count().alias("Count"),
    )
    return


@app.cell
def _(df, pl):
    df.filter((pl.col("x") == 0) | (pl.col("y") == 0) | (pl.col("z") == 0))
    #added to filter
    return


@app.cell
def _(alt, cs, df, pl):
    corr = df.select(cs.numeric()).corr()

    corr_long = (
        corr
        .with_columns(column=pl.Series(corr.columns)) # Add a column to identify the rows
        .unpivot(index="column")                    # Melt the data (wide to long)
        .rename({"column": "Variable 1", "variable": "Variable 2", "value": "Correlation"})
    )

    base = alt.Chart(corr_long).encode(
        x='Variable 1:O',
        y='Variable 2:O'
    )

    # The Heatmap Square
    heatmap = base.mark_rect().encode(
        color=alt.Color('Correlation:Q', 
                        scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                        title="Pearson Correlation")
    )

    # The Correlation Text Labels
    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('Correlation:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Correlation > 0.5, 
            alt.value('white'), 
            alt.value('black')
        )
    )

    # Combine and configure
    chart = (heatmap + text).properties(
        width=400,
        height=400,
        title="Correlation Matrix"
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    chart
    return


if __name__ == "__main__":
    app.run()
