# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output, ClientsideFunction
import plotly.express as px
import pandas as pd
import pathlib
import plotly.graph_objects as go
from plotly.graph_objects import Layout
import numpy as np
import xarray as xr


app = Dash(__name__)

app.title = 'DSI Dashboard'

##import data
# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.resolve()

# Read data
model_table = pd.read_csv(DATA_PATH.joinpath("model_table.csv"))

##preparation for choropleth
map_df = model_table.groupby('PHYSICIAN_STATE')['TARGET'].apply(lambda x: (x > 0).sum()).reset_index(name='SUM_TARGET')
map_df_per = model_table.groupby('PHYSICIAN_STATE')['PATIENT_ID'].count().reset_index(name='PATIENT_COUNT')
map_df = pd.merge(left=map_df, right=map_df_per, how='left', left_on='PHYSICIAN_STATE', right_on='PHYSICIAN_STATE')
map_df['PERCENTAGE'] = ((map_df['SUM_TARGET']*100)/map_df['PATIENT_COUNT']).round(decimals = 2)


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("DSI Analytics"),
            html.H3("Welcome to the DSI Dashboard"),
            html.Div(
                id="intro",
                    children="Drug A is an oral antiviral that can be used to treat mild to moderate infections of Disease X in patients over the age of 12 who are at high risk of progressing to severe Disease X. Drug A must be taken within 5 days of showing symptoms caused by a Disease X infection.",
            ),
            html.Div(
                id="intro2",
                children="The Data Science Team prepared a model_table to help the Drug A Team understand what factors make patients more likely to receive treatment following a diagnosis of Disease X through Machine Learning.",
            ),
            html.Div(
                id="intro3",
                children="This dashboard allows to explore and analyze the model_table data.",
            ),
        ],
    )


layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

fig_choro = px.choropleth(
    map_df,
    locations = 'PHYSICIAN_STATE',
    locationmode = 'USA-states',
    color = 'SUM_TARGET',
    scope = 'usa',
    color_continuous_scale = 'Blues',
    title = 'Drug A prescribed per state',
    hover_data = ['PATIENT_COUNT', 'PERCENTAGE'])

fig_gen = px.histogram(model_table, x='PATIENT_GENDER', y='PATIENT_ID',
             color='TARGET', barmode='group',
             height=400, histfunc='count', color_discrete_map={1:'#3888c1', 0:'#b9d6e6'})

fig_gen.update_layout(title='Patient gender and Drug A prescription', 
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend_title='Target',
                      yaxis=dict(
                          title='Patients',
                          titlefont_size=14,
                          tickfont_size=14),
                      xaxis=dict(
                          title='Patient gender',
                          titlefont_size=14,
                          tickfont_size=14))

t_1 = model_table.loc[model_table['TARGET'] == 1]
t_0 = model_table.loc[model_table['TARGET'] == 0]

counts1, bins1 = np.histogram(t_1.PATIENT_AGE, bins=range(0, 100, 10))
bins1 = 0.5 * (bins1[:-1] + bins1[1:])

counts0, bins0 = np.histogram(t_0.PATIENT_AGE, bins=range(0, 100, 10))
bins0 = 0.5 * (bins0[:-1] + bins0[1:])


fig_age = go.Figure(layout=layout)
fig_age.add_trace(go.Bar(name='0', x=bins0, y=counts0, marker_color='#b9d6e6'))
fig_age.add_trace(go.Bar(name='1', x=bins1, y=counts1, marker_color='#3888c1'))


fig_age.update_layout(title='Age of patients prescribed and not prescribed with drug A', 
        yaxis=dict(
        title='Patients',
        titlefont_size=14,
        tickfont_size=14),
        xaxis=dict(
        title='Patient age when diagnosed',
        titlefont_size=14,
        tickfont_size=14),
        legend_title='Target',
)

#Create Age Bins
bins= [0,18,35,45,60,100]
labels = ['0-18','18-35','35-45','45-60','60-100']
model_table['AGE_GROUP'] = pd.cut(model_table.PATIENT_AGE, bins=bins, labels=labels, right=False)

# Create dimensions
class_dim = go.parcats.Dimension(
    values=model_table.AGE_GROUP,
    categoryorder='category ascending', label='Age'
)

gender_dim = go.parcats.Dimension(values=model_table.PATIENT_GENDER, label='Gender')

survival_dim = go.parcats.Dimension(
    values=model_table.TARGET, label='Target', categoryarray=[0, 1],
    ticktext=['No Drug A', 'Drug A']
)

# Create parcats trace
color = model_table.TARGET;
colorscale = [[0, '#b9d6e6'], [1, '#3888c1']];

fig_san = go.Figure(data = [go.Parcats(dimensions=[class_dim, gender_dim, survival_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', hoverinfo='count+probability',
        arrangement='freeform')])

fig_san.update_layout(title='Target categorization')


fig_corr = px.imshow(model_table.corr(), color_continuous_scale='RdBu_r', origin='lower')

fig_corr.update_layout(title='Model_table correlation plot')


fig_type= px.histogram(model_table, y='PHYSICIAN_TYPE', x='PATIENT_ID',
             color='TARGET', barmode='group', histfunc='count',
             height=700, color_discrete_map={1: '#3888c1', 0:'#b9d6e6'}).update_yaxes(categoryorder='total ascending')

fig_type.update_layout(title='Physician types and drug A prescription',
                       plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                        title='Patients',
                        titlefont_size=14,
                        tickfont_size=14),
                        legend_title='Target',)

app.layout = html.Div(
    id='app-container',
    children=[
        #banner
        html.Div(
            id="banner",
            className="banner",
            #children=[html.Img(src=app.get_asset_url(""))],
        ),
        #short description
        #left col
        html.Div(
            id='left-col',
            className = 'three columns',
            children=[description_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        #center col
        html.Div(
            id='center-col',
            className='five columns',
            children=[
                #choropleth
                html.Div(
                    id='choropleth_map',
                    children=[
                        html.Hr(),
                        dcc.Graph(
                        id='choro-graph',
                        figure=fig_choro
                        ),
                    ],   
                ),
                #gender plot
                html.Div(
                    id='gender_plot',
                    children=[
                        html.Hr(),
                        dcc.Graph(
                        id='gender-graph',
                        figure=fig_gen
                        ),
                    ],   
                ),
                #phy type plot
                html.Div(
                    id='type_plot',
                    children=[
                        html.Hr(),
                        dcc.Graph(
                        id='type-graph',
                        figure=fig_type
                        ),
                    ],   
                ),
            ]
        ),
        #right col
         html.Div(
            id='right-col',
            className='four columns',
            children=[
                #choropleth
                html.Div(
                    id='age_barchart',
                    children=[
                        html.Hr(),
                        dcc.Graph(
                        id='age-graph',
                        figure=fig_age
                        ),
                    ],   
                ),
                #gender plot
                html.Div(
                    id='sankey_plot',
                    children=[
                        html.Hr(),
                        dcc.Graph(
                        id='sankey-graph',
                        figure=fig_san
                        ),
                    ],   
                ),
                #correlation plot
                html.Div(
                    id='corr_plot',
                    children=[
                        html.Hr(),
                        dcc.Graph(
                        id='corr-graph',
                        figure=fig_corr
                        ),
                    ],   
                ),
            ]
        ),
])

if __name__ == '__main__':
    app.run_server(debug=True)