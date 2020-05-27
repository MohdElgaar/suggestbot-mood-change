import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from datetime import date, time, datetime

app = dash.Dash(__name__)

placeholder_data = pd.DataFrame({'action': ['Facebook (App)', 'Call',
    'Weather Change', 'Exercise', 'Youtube (App)', 'Netflix (App)', 'Extra'],
    'type': ['Pos', 'Pos', 'Pos', 'Neg', 'Neg', 'Neg', 'Pos'],
    'time': [time(13, 00), time(15, 0), time(12, 0), time(11, 0),
        time(19, 0), time(17, 0), None]})

placeholder_data_line = pd.DataFrame({
    'time': [time(i) for i in range(9,22)],
    'valence': [3, 5, 6, 7, 4, 2, -1, -4, -2, 0, 2, 3, 3],
    'UV': [0.5, 0.5, 1, 5, 4, 5, 0.5, 0.5, 0.5, 3, 0, 0, 0]})

def fill_table(df):
    pos = df[df['type'] == 'Pos']
    neg = df[df['type'] == 'Neg']

    maxlen = min(len(pos), len(neg))

    rows = []
    for p,n in zip(pos['action'], neg['action']):
        rows.append(html.Tr([html.Td([p]), html.Td([n])]))

    if len(pos) > maxlen:
        for p in pos[maxlen:]['action']:
            rows.append(html.Tr([html.Td([p]), ""]))
    elif len(neg) > maxlen:
        for n in neg[maxlen:]['action']:
            rows.append(html.Tr([html.Td([p]), ""]))

    return rows

def line_plot(df_line, df_data, UV = False):
    valence = go.Scatter(x = df_line['time'], y = df_line['valence'],
            name = "Valence")
    
    layout = {'images':
                    [{"x": (13-9)/(21-9), "y": (7+5-4+1)/(7+5),
                        'sizex': 0.1, 'sizey': 0.1,
                        'source': "/assets/fb.png",
                        'xanchor': "left",
                      'xref': "paper",
                      'yanchor': "center",
                      'yref': "paper"}]}

    if UV:
        UV_exposure = go.Scatter(x = df_line['time'], y = df_line['UV'], name = "UV")
        return {'data': [valence, UV_exposure],
                'layout': layout}
    else:
        return {'data': [valence], layout: layout}

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
    html.Div(className = 'left', children = [
        html.H1("Welcome"),
        dcc.Graph(id = 'graph',
            figure = line_plot(placeholder_data_line, placeholder_data,
                UV=True))
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
