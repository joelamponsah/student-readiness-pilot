import plotly.express as px

def line_plot(df, x, y, color=None, title=""):
    return px.line(df, x=x, y=y, color=color, markers=True, title=title)

def bar_plot(df, x, y, color=None, title=""):
    return px.bar(df, x=x, y=y, color=color, title=title)
