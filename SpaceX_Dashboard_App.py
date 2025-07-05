# Import required libraries
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
spacex_df = pd.read_csv(url)
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                dcc.Dropdown(id='site-dropdown', 
                                                    options=[{'label': 'All Sites', 'value': 'ALL'}] + 
                                                    [{'label': site, 'value': site} for site in spacex_df['Launch Site'].unique()],
                                                    value='ALL',
                                                    placeholder='Select Launch Site',
                                                    searchable=True
                                                    ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload Range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(
                                    id='payload-slider',
                                    min=0,
                                    max=10000,
                                    step=100,
                                    marks={0: '0', 2500: '2500', 5000: '5000', 7500: '7500', 10000: '10000'},
                                    value=[min_payload, max_payload]),
                                html.Div(id='slider-output-container'),
                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output

@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    Output(component_id='slider-output-container', component_property='children'),
    Input(component_id='site-dropdown', component_property='value'),
    Input(component_id='payload-slider', component_property='value')
)
def update_charts(entered_site, payload_range):
    # Pie chart
    if entered_site == 'ALL':
        pie_fig = px.pie(
            spacex_df, 
            values='class', 
            names='Launch Site', 
            title='Total Success Launches by Site'
        )
    else:
        site_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        outcome_counts = site_df['class'].value_counts().reset_index()
        outcome_counts.columns = ['Outcome', 'Count']
        outcome_counts['Outcome'] = outcome_counts['Outcome'].map({1: 'Success', 0: 'Failure'})
        pie_fig = px.pie(
            outcome_counts,
            values='Count',
            names='Outcome',
            title=f'Total Success vs Failure for site {entered_site}'
        )

    # Scatter chart
    filtered_df = spacex_df[
        (spacex_df['Payload Mass (kg)'] >= payload_range[0]) &
        (spacex_df['Payload Mass (kg)'] <= payload_range[1])
    ]
    if entered_site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
    scatter_fig = px.scatter(
        filtered_df, x='Payload Mass (kg)', y='class',
        color='Booster Version Category',
        title='Correlation between Payload and Success for ' + (entered_site if entered_site != 'ALL' else 'All Sites'),
        labels={'class': 'Launch Outcome'}
    )
    slider_text = f"Selected payload range: {payload_range[0]} kg - {payload_range[1]} kg"
    return pie_fig, scatter_fig, slider_text

# Run the app
if __name__ == '__main__':
    app.run()
