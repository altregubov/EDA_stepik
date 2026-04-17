# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "marimo>=0.21.1",
#     "numpy==2.4.3",
#     "polars==1.39.3",
#     "scikit-learn==1.8.0",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs
    import altair as alt
    import vegafusion as vf
    alt.data_transformers.enable("vegafusion")

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV


    return (
        GaussianNB,
        GridSearchCV,
        KNeighborsClassifier,
        Pipeline,
        StandardScaler,
        alt,
        pl,
        train_test_split,
    )


@app.cell
def _(pl):
    df = pl.read_parquet("./diamonds.parquet")
    df
    return (df,)


@app.cell
def _(alt, df):
    alt.Chart(df, width=400, height=400).mark_bar().encode(
        x=alt.X("price:Q").bin(maxbins=100),
        y='count()',
    )
    return


@app.cell
def _(df, pl):
    X = df.select(['carat','depth','table',"x","y","z"])
    y = df.select(
        y_class = (pl.col("price") > 2500).cast(pl.Int32)
    )


    y
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    GaussianNB,
    GridSearchCV,
    KNeighborsClassifier,
    Pipeline,
    StandardScaler,
    X_test,
    X_train,
    y_test,
    y_train,
):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB()) # По умолчанию поставим Байес
    ])

    # Описываем сетку параметров для обоих алгоритмов
    param_grid = [
        {
            'classifier': [GaussianNB()],
            # У Гауссовского Байеса почти нет гиперпараметров для настройки
        },
        {
            'classifier': [KNeighborsClassifier()],
            'classifier__n_neighbors': [3, 5, 7, 10],
            'classifier__weights': ['uniform', 'distance']
        }
    ]

    # Настраиваем поиск
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print(f"Лучшая модель: {grid.best_params_}")
    print(f"Лучшая точность на кросс-валидации: {grid.best_score_:.4f}")
    print(f"Точность на тестовых данных: {grid.score(X_test, y_test):.4f}")
    return (grid,)


@app.cell
def _(grid, pl):
    df_results = pl.DataFrame(grid.cv_results_, strict=False)
    return (df_results,)


@app.cell
def _(df_results):
    df_results
    return


if __name__ == "__main__":
    app.run()
