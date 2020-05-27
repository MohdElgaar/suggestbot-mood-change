import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import date, time, datetime

app = dash.Dash(__name__)

placeholder_data = pd.DataFrame({'action': ['Facebook (App)', 'Call',
    'Weather Change', 'Exercise', 'Youtube (App)', 'Netflix (App)', 'Extra'],
    'type': ['Pos', 'Pos', 'Pos', 'Neg', 'Neg', 'Neg', 'Pos'],
    'time': [time(13, 00), time(15, 0), time(12, 0), time(11, 0),
        time(19, 0), time(17, 0), None],
    'icon': ['fb.png', 'phone.png', 'weather.png', 'workout.png',
        'yt.png', 'netflix.png', None]})

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

    df_data.dropna(subset=['time'], inplace=True)
    max_time = 21
    min_time = 9
    max_valence = max(df_line['valence'])
    min_valence = -5
    
    layout = {'images': [],
            'paper_bgcolor': 'lightgray'}
    for _, item in df_data.iterrows():
        cur_valence = df_line[df_line['time'] == item['time']].iloc[0]['valence']
        if item['icon']:
            layout['images'].append({
                "x": (item['time'].hour-min_time)/\
                        (max_time-min_time),
                "y": (cur_valence-min_valence)/\
                        (max_valence - min_valence + 1),
                'sizex': 0.08, 'sizey': 0.08,
                'source': "/assets/%s"%item['icon'],
                'xanchor': "left",
                'xref': "paper",
                'yanchor': "center",
                'yref': "paper"})

            layout['images'].append({
                "x": (item['time'].hour-min_time)/\
                        (max_time-min_time),
                "y": (cur_valence-min_valence)/\
                        (max_valence - min_valence + 1),
                'sizex': 0.08, 'sizey': 0.08,
                'source': "/assets/red.png" if item['type'] == 'Pos' else\
                        "/assets/green.png",
                'xanchor': "left",
                'xref': "paper",
                'yanchor': "center",
                'yref': "paper",
                'opacity': 0.3})

    if UV:
        UV_exposure = go.Scatter(x = df_line['time'], y = df_line['UV'], name = "UV")
        return {'data': [valence, UV_exposure],
                'layout': layout}
    else:
        return {'data': [valence], 'layout': layout}

app.layout = html.Div(id='bottom', children = [
    html.Div(className = 'right', children = [
        html.Table(className = 'content-table', children = [
            html.Thead([
                html.Tr([
                    # html.Th(id='pos', children = ["Positive"]),
                    html.Th([
                    html.Button("Positive", id = 'pos'),
                    ]),
                    # html.Th(id='neg', children = ["Negative"])
                    html.Th([
                    html.Button("Negative", id = 'neg')
                    ]),
                ])
                ]),
            html.Tbody(fill_table(placeholder_data))
            ])
        ]),
    html.Div(className = 'left', children = [
        html.H1("Positive & Negative Mood Influencers"),
        html.Div(id='plot', children=[dcc.Graph(id = 'line_plot')]),
        html.Center([html.Button("Show UV Exposure", id = "UV_button",
            n_clicks = 0)])
        ])
    ])

@app.callback(
        Output("line_plot", 'figure'), [Input("UV_button", "n_clicks")]
        )
def callback(n_clicks):
    if n_clicks % 2 == 0:
        return line_plot(placeholder_data_line, placeholder_data, UV=False)
    else:
        return line_plot(placeholder_data_line, placeholder_data, UV=True)

if __name__ == '__main__':
    app.run_server(debug=True)
