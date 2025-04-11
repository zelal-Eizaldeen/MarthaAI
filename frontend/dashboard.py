import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px

PATH_TO_DATA='../data'
sample_ads=pd.read_csv(f'{PATH_TO_DATA}/meta_data.csv')

# Load the data of RMSE
rmse_ads=pd.read_csv(f'{PATH_TO_DATA}/rf_predictions_with_rmse.csv')
# Extract RMSE from the last row
rmse_row = rmse_ads.iloc[-1]
if rmse_row['Actual_Conversions'] == 'RMSE':
    rmse = float(rmse_row['Predicted_Conversions'])
    predictions_df = rmse_ads.iloc[:-1]  # Remove RMSE row from main table
else:
    rmse = None  # Handle case where RMSE row is missing
    predictions_df = rmse_ads  # fallback

# Load the data of Features importance
feature_df=pd.read_csv(f'{PATH_TO_DATA}/rf_feature_importance.csv')


# Dash App
# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Meta Ads + ML Dashboard"

# Layout
app.layout = html.Div([
    html.H1("\U0001F4C8 Meta Ads Dashboard (w/ ML Insights)", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Campaign(s):"),

        dcc.Dropdown(
            options=[{'label': c, 'value': c} for c in sample_ads['campaign'].unique()],
            value=[],
            multi=True,
            id='campaign-filter'
        )
    ], style={'width': '40%', 'margin': 'auto'}),

    html.Div(id='metrics', style={'textAlign': 'center', 'marginTop': 30}),
    dcc.Graph(id='sales-trend-graph'),
    html.Div(id='alert-box', style={'textAlign': 'center', 'color': 'red', 'fontWeight': 'bold', 'marginTop': 20}),

    html.H2("\U0001F916 ML Model Insights", style={'textAlign': 'center', 'marginTop': 50}),
    html.P(f"Root Mean Squared Error (RMSE): {rmse:.2f}", style={'textAlign': 'center', 'fontSize': 18}),

    html.H3("Feature Importance", style={'textAlign': 'center'}),
    dcc.Graph(
        figure=px.bar(
            feature_df,
            x='Feature',
            y='Importance',
            title="Top Features Influencing Ad Conversions",
            labels={"Importance": "Importance Score", "Feature": "Feature"}
        )
    ),

    html.H3("Prediction Results Table", style={'textAlign': 'center'}),
    dash_table.DataTable(
        data=predictions_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in predictions_df.columns],
        page_size=10,
        style_table={'margin': 'auto', 'width': '70%'},
        style_cell={'textAlign': 'center'}
    )
])

# Callback for updating metrics and sales trend
date_format = '%Y-%m-%d'

@app.callback(
    [Output('metrics', 'children'),
     Output('sales-trend-graph', 'figure'),
     Output('alert-box', 'children')],
    [Input('campaign-filter', 'value')]
)
def update_dashboard(selected_campaigns):
    df = sample_ads.copy()
    if selected_campaigns:
        df = df[df['campaign'].isin(selected_campaigns)]

    df['date'] = pd.to_datetime(df['date'])
    total_revenue = df['ad_spend'].sum()
    avg_ctr = df['ctr'].mean()
    total_conversions = df['conversions'].sum()

    trend = df.set_index('date').resample('D')['conversions'].sum()
    fig = px.line(trend, title='\U0001F4CA Daily Conversions Over Time', labels={'value': 'Conversions', 'date': 'Date'})

    mean_conv = trend.mean()
    std_conv = trend.std()
    alert = ""
    if len(trend) > 0 and trend.iloc[-1] > mean_conv + 2 * std_conv:
        alert = "\U0001F6A8 Conversion spike detected – check campaign details!"
    elif len(trend) > 0 and trend.iloc[-1] < mean_conv - 2 * std_conv:
        alert = "\u26A0\uFE0F Drop in conversions – review ad creative & targeting."

    metrics = html.Div([
        html.H3(f"\U0001F4B8 Total Ad Spend: ${total_revenue:,.2f}"),
        html.H4(f"\U0001F501 Avg CTR: {avg_ctr:.2%}"),
        html.H4(f"\U0001F3AF Total Conversions: {total_conversions}")
    ])

    return metrics, fig, alert

# Run app
if __name__ == '__main__':
    app.run(debug=True)