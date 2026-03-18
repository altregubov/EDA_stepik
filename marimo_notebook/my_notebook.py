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
def _(pl):
    df = pl.read_csv("https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/diamonds_good.csv")

    rename_mapping = {
        "'x'": "x",
        "'y'": "y",
        "'z'": "z",
    }

    df = df.rename(rename_mapping)
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(alt, cs, df):
    alt.Chart(df, width=200, height=200).mark_bar().encode(
        x=alt.X(alt.repeat('repeat'), type='quantitative', bin=alt.Bin(maxbins=100)),
        y='count()',
    ).repeat(
        repeat=df.select(cs.numeric()).columns,
        columns=3
    )
    return


if __name__ == "__main__":
    app.run()
