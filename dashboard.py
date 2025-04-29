# dashboard.py
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend first
import nltk
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud
import base64
from io import BytesIO
import os
import sys
import google.generativeai as genai
from nltk.sentiment.vader import SentimentIntensityAnalyzer
URL = "https://www.flipkart.com/portronics-key7-combo-keyboard-mouse-set-2-4ghz-1200-dpi-wireless-standard-laptop-compatible-desktop-laptop-mac/p/itmd19f09775a91e?pid=ACCGRY7J9NRSWAZB&lid=LSTACCGRY7J9NRSWAZBWZS1HP&marketplace=FLIPKART&q=Portronics+Key7+Combo+Keyboard+and+Mouse+Set+Summary+Based+on+Reviews%3A&store=6bo%2Fai3%2F3oe&srno=s_1_1&otracker=search&otracker1=search&fm=Search&iid=en_A08W-aH0I6GuQEeWYfx8-DaDOdvbPuRLDdIyEloiGPQJMLLF2L3pTAl6wec6Y3J4wzVluBgyUetsai_RBuXF4PUFjCTyOHoHZs-Z5_PS_w0%3D&ppt=sp&ppn=sp&ssid=2bnzpqmsj1hy2osg1745089575312&qH=658e9ee1efbab613"
# Try to download nltk data if needed
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Create a simple way to directly load the dataframe without running the analysis code
# This assumes that your analysis.py file sets a global variable named 'df'
# You should extract your dataframe creation logic to a separate file or function
# For now, as a workaround, we'll create a mock dataframe with expected structure

# OPTION 1: Create a mock dataframe for demonstration (use this if you can't modify analysis.py)
# Uncomment this code if you want to use a mock dataframe
"""
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'text': ['This is a sample review text ' * 3] * 100,
    'rating': [1, 2, 3, 4, 5] * 20,
    'sentiment_score': [0.8, -0.5, 0.9, 0.2, -0.7] * 20,
    'sentiment_label': ['POSITIVE', 'NEGATIVE'] * 50,
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'] * 20,
    'certified_buyer': [True, False] * 50
})
"""

# OPTION 2: Try to import the dataframe directly (safer approach)
# This requires minor modification to analysis.py

# First, check if we've already imported the df to avoid reloading
if 'df' not in globals():
    # We need to safely import df from analysis.py
    try:
        # Try to import as a module (if correctly structured)
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from analysis import df
    except:
        # If that doesn't work, read the CSV directly if available
        try:
            # Assuming your data is in a CSV file, adjust the path as needed
            df = pd.read_csv('reviews.csv')
        except:
            # Last resort - create a mock dataset
            print("Warning: Could not load data from analysis.py. Using mock data.")
            df = pd.DataFrame({
                'date': [f"Jan, {year}" for year in range(2020, 2025) for _ in range(20)],
                'text': ['This is a sample review text ' * 3] * 100,
                'rating': [1, 2, 3, 4, 5] * 20,
                'sentiment_score': [0.8, -0.5, 0.9, 0.2, -0.7] * 20,
                'sentiment_label': ['POSITIVE', 'NEGATIVE'] * 50,
                'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'] * 20,
                'certified_buyer': [True, False] * 50
            })

# Initialize Dash app with responsive settings
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.MATERIA],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True)
app.title = "Advanced Review Analytics Dashboard"

# Preprocess data - convert dates to datetime objects
try:
    df['date'] = pd.to_datetime(df['date'], format='%b, %Y')
except:
    # If date conversion fails with the specified format, try auto-detection
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])

# Initialize Gemini API (Replace with your API key)
GEMINI_API_KEY = "YOUR API KEY"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Configure the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Add fake review column (since it's missing in the data)
# Using a simple approach to simulate fake review detection
# In a real system, you would use the detectFakeReview function
# instead of this random assignment
import numpy as np
np.random.seed(42)  # For reproducibility
df['is_fake'] = np.random.choice(['real', 'fake'], size=len(df), p=[0.8, 0.2])

# Helper functions
def detect_fake_review(text):
    """Use Gemini to detect fake reviews"""
    try:
        prompt = f"""Analyze this review for authenticity. Respond ONLY with 'real' or 'fake':
        
        Review: {text}"""
        
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        
        # Ensure we only return 'real' or 'fake'
        if 'fake' in result:
            return 'fake'
        else:
            return 'real'
    except Exception as e:
        print(f"Error in fake review detection: {e}")
        return "real"  # Default to real if API call fails

def generate_product_summary(reviews):
    """Generate product summary using Gemini"""
    try:
        global URL
        prompt = f"""Consider the product at URL: {URL}
        Generate a comprehensive product summary based on these reviews. 
        Highlight key features, common complaints, and overall sentiment. Use bullet points.
        
        Reviews:\n{"\n".join(reviews)}"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Summary unavailable - API call failed. Please check your Gemini API key or connection."

def plot_wordcloud(text_series):
    if text_series.empty:
        return None
    text = ' '.join(text_series.dropna().tolist())
    wc = WordCloud(width=800, height=400, background_color='white', prefer_horizontal=0.8)
    wc.generate(text)
    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def create_filtered_df(start_date, end_date, cities, certified):
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['date'] >= start_date) & 
                             (filtered_df['date'] <= end_date)]
    if cities:
        filtered_df = filtered_df[filtered_df['city'].isin(cities)]
    if certified and 'certified' in certified:
        filtered_df = filtered_df[filtered_df['certified_buyer']]
    return filtered_df

# Create empty figure for when no data is available
def empty_figure(title='No Data Available'):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="No data available for the selected filters",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
        ]
    )
    return fig

# Update layout with new components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Advanced Review Analytics", className="text-center mb-4 h3"), width=12)
    ]),
    
    # KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Reviews", className="card-title"),
                html.H4(id='total-reviews', className="card-text")
            ])
        ], className="h-100"), width=6, md=3, lg=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Avg. Rating", className="card-title"),
                html.H4(id='avg-rating', className="card-text")
            ])
        ], className="h-100"), width=6, md=3, lg=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Positive %", className="card-title"),
                html.H4(id='positive-pct', className="card-text")
            ])
        ], className="h-100"), width=6, md=3, lg=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Certified %", className="card-title"),
                html.H4(id='certified-pct', className="card-text")
            ])
        ], className="h-100"), width=6, md=3, lg=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Fake Reviews", className="card-title"),
                html.H4(id='fake-reviews', className="card-text")
            ])
        ], className="h-100"), width=6, md=3, lg=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Fake %", className="card-title"),
                html.H4(id='fake-pct', className="card-text")
            ])
        ], className="h-100"), width=6, md=3, lg=2)
    ], className="g-2 mb-4"),
    
    # Filters & Content
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Filters", className="mb-3"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=df['date'].min(),
                    end_date=df['date'].max(),
                    display_format='MMM YYYY'
                ),
                html.Hr(),
                dcc.Dropdown(
                    id='city-filter',
                    options=[{'label': c, 'value': c} for c in df['city'].unique()],
                    multi=True,
                    placeholder="All Cities"
                ),
                html.Hr(),
                dcc.Checklist(
                    id='certified-filter',
                    options=[{'label': ' Certified Buyers', 'value': 'certified'}],
                    className="ps-2"
                )
            ], className="bg-light p-3 rounded-3")
        ], width=12, lg=2, className="mb-4 mb-lg-0"),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='sentiment-distribution'), width=6),
                dbc.Col(dcc.Graph(id='fake-analysis'), width=6)
            ], className="g-3 mb-4"),
            
            dbc.Row([
                dbc.Col(dcc.Markdown(id='product-summary', 
                                   children="",
                                   className="border p-3 rounded-3"),
                      width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='rating-analysis'), width=6),
                dbc.Col(dcc.Graph(id='sentiment-over-time'), width=6)
            ], className="g-3 mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.RadioItems(
                            id='wordcloud-selector',
                            options=['Positive', 'Negative'],
                            value='Positive',
                            inline=True,
                            className="mb-3 justify-content-center d-flex"
                        ),
                        html.Img(id='wordcloud-image', className="img-fluid")
                    ], className="text-center border p-3 rounded-3")
                ], width=6),
                
                dbc.Col(dcc.Graph(id='certified-impact'), width=6)
            ], className="g-3")
        ], width=12, lg=10)
    ], className="g-3")
], fluid=True, className="py-3")

# Updated Callbacks with error handling
@app.callback(
    [Output('total-reviews', 'children'),
     Output('avg-rating', 'children'),
     Output('positive-pct', 'children'),
     Output('certified-pct', 'children'),
     Output('fake-reviews', 'children'),
     Output('fake-pct', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('city-filter', 'value'),
     Input('certified-filter', 'value')]
)
def update_kpis(start_date, end_date, cities, certified):
    filtered_df = create_filtered_df(start_date, end_date, cities, certified)
    total = len(filtered_df)
    avg_rating = f"{filtered_df['rating'].mean():.2f}" if total > 0 else "N/A"
    positive_pct = f"{(filtered_df['sentiment_label'].value_counts(normalize=True).get('POSITIVE', 0)*100):.1f}%" 
    certified_pct = f"{(filtered_df['certified_buyer'].mean()*100):.1f}%"
    
    # Calculate fake reviews metrics
    fake_count = filtered_df[filtered_df['is_fake'] == 'fake'].shape[0]
    fake_pct = f"{(fake_count/total*100):.1f}%" if total > 0 else "N/A"
    
    return total, avg_rating, positive_pct, certified_pct, fake_count, fake_pct

@app.callback(
    Output('sentiment-distribution', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('city-filter', 'value'),
     Input('certified-filter', 'value')]
)
def update_sentiment_distribution(start_date, end_date, cities, certified):
    filtered_df = create_filtered_df(start_date, end_date, cities, certified)
    
    if filtered_df.empty:
        return empty_figure("No Sentiment Data Available")
    
    # Create counts manually
    sentiment_counts = filtered_df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    # Create a simple pie chart with go instead of px
    colors = {'POSITIVE': '#2E7D32', 'NEGATIVE': '#C62828'}
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts['sentiment'],
        values=sentiment_counts['count'],
        hole=0.4,
        marker_colors=[colors.get(val, '#888888') for val in sentiment_counts['sentiment']]
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=False
    )
    
    return fig

@app.callback(
    Output('rating-analysis', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('city-filter', 'value'),
     Input('certified-filter', 'value')]
)
def update_rating_analysis(start_date, end_date, cities, certified):
    filtered_df = create_filtered_df(start_date, end_date, cities, certified)
    
    if filtered_df.empty:
        return empty_figure("No Rating Data Available")
    
    # Group by rating and sentiment manually
    rating_counts = filtered_df.groupby(['rating', 'sentiment_label']).size().reset_index(name='count')
    
    # Create figure using go.Bar instead of px.bar
    fig = go.Figure()
    
    # Add traces for each sentiment
    for sentiment in rating_counts['sentiment_label'].unique():
        df_sentiment = rating_counts[rating_counts['sentiment_label'] == sentiment]
        color = '#2E7D32' if sentiment == 'POSITIVE' else '#C62828'
        
        fig.add_trace(go.Bar(
            x=df_sentiment['rating'],
            y=df_sentiment['count'],
            name=sentiment,
            marker_color=color
        ))
    
    fig.update_layout(
        title='Rating Distribution by Sentiment',
        xaxis_title='Rating',
        yaxis_title='Count',
        barmode='stack'
    )
    
    return fig

@app.callback(
    Output('sentiment-over-time', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('city-filter', 'value'),
     Input('certified-filter', 'value')]
)
def update_sentiment_over_time(start_date, end_date, cities, certified):
    filtered_df = create_filtered_df(start_date, end_date, cities, certified)
    
    if filtered_df.empty:
        return empty_figure("No Time-Series Data Available")
    
    # Resample data by month
    time_series = filtered_df.set_index('date')['sentiment_score'].resample('ME').mean().reset_index()
    
    # Create line chart with go.Scatter
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_series['date'],
        y=time_series['sentiment_score'],
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title='Sentiment Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-1, 1])
    )
    
    return fig

@app.callback(
    Output('certified-impact', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('city-filter', 'value'),
     Input('certified-filter', 'value')]
)
def update_certified_impact(start_date, end_date, cities, certified):
    filtered_df = create_filtered_df(start_date, end_date, cities, certified)
    
    if filtered_df.empty:
        return empty_figure("No Certification Data Available")
    
    # Group by certified_buyer
    certified_stats = filtered_df.groupby('certified_buyer')['sentiment_score'].mean().reset_index()
    
    # Create bar chart with go.Bar
    fig = go.Figure()
    
    colors = [('#2E7D32' if is_cert else '#C62828') for is_cert in certified_stats['certified_buyer']]
    
    fig.add_trace(go.Bar(
        x=certified_stats['certified_buyer'].map({True: 'Certified', False: 'Not Certified'}),
        y=certified_stats['sentiment_score'],
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Sentiment by Certified Buyer',
        xaxis_title='Certified Buyer',
        yaxis_title='Avg. Sentiment Score'
    )
    
    return fig

@app.callback(
    Output('fake-analysis', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('city-filter', 'value'),
     Input('certified-filter', 'value')]
)
def update_fake_analysis(start_date, end_date, cities, certified):
    filtered_df = create_filtered_df(start_date, end_date, cities, certified)
    
    if filtered_df.empty:
        return empty_figure("No Authenticity Data Available")
    
    # Get the fake review distribution
    fake_counts = filtered_df['is_fake'].value_counts().reset_index()
    fake_counts.columns = ['authenticity', 'count']
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=fake_counts['authenticity'],
        values=fake_counts['count'],
        hole=0.4,
        marker_colors=['#4CAF50' if auth == 'real' else '#F44336' for auth in fake_counts['authenticity']]
    )])
    
    fig.update_layout(
        title="Review Authenticity",
        showlegend=True
    )
    
    return fig

@app.callback(
    Output('product-summary', 'children'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('city-filter', 'value'),
    Input('certified-filter', 'value')
)
def update_summary(start_date, end_date, cities, certified):
    filtered = create_filtered_df(start_date, end_date, cities, certified)
    
    if filtered.empty:
        return "No reviews available for summary generation."
    
    reviews = filtered['text'].sample(min(10, len(filtered))).tolist()
    if not reviews:
        return "No reviews available for summary generation."
    
    try:
        return generate_product_summary(reviews)
    except Exception as e:
        return f"Error generating summary: {str(e)}. Please check your Gemini API key or connection."

@app.callback(
    Output('wordcloud-image', 'src'),
    [Input('wordcloud-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('city-filter', 'value'),
     Input('certified-filter', 'value')]
)
def update_wordcloud(selector, start_date, end_date, cities, certified):
    filtered_df = create_filtered_df(start_date, end_date, cities, certified)
    
    if filtered_df.empty:
        return ''
    
    text_series = filtered_df[filtered_df['sentiment_label'] == selector.upper()]['text']
    
    if text_series.empty:
        return ''
    
    try:
        img_src = plot_wordcloud(text_series)
        return f"data:image/png;base64,{img_src}" if img_src else ''
    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return ''

if __name__ == '__main__':
    app.run(debug=False, dev_tools_props_check=False)
