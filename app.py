import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from datetime import date, time, datetime

app = dash.Dash(__name__)

placeholder_data = None
app.layout = html.Div([
    html.Div(className = 'right', children = [
        html.Table([html.Thead(
            [html.Tr(
                [html.Th(["Positive"]),
                html.Th(["Negative"])
                ])
                ]),
            html.Tbody(fill_table(placeholder_data))
            ])
        ]),
    html.Div(className = 'left', children = [html.H1("Welcome")])
    ])

def fill_table(df):
    pass

if __name__ == '__main__':
    app.run_server(debug=True)
