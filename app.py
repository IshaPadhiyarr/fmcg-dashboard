# =================== IMPORTS ===================
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import re
import os

import plotly.express as px
import plotly.graph_objs as go

from dash import Dash, html, dcc, Input, Output

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from wordcloud import WordCloud

# =================== FILE PATHS (relative - works anywhere) ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMPAIGN_RESPONSE_PATH = os.path.join(BASE_DIR, "data", "Campaign_Response__Data.csv")
CAMPAIGN_DETAILS_PATH  = os.path.join(BASE_DIR, "data", "Campaign_Details.csv")
MASTER_LOOKUP_PATH     = os.path.join(BASE_DIR, "data", "MasterLookUp.csv")
TRANSACTION_DATA_PATH  = os.path.join(BASE_DIR, "data", "Transaction_data.csv")
REVIEWS_PATH           = os.path.join(BASE_DIR, "data", "74responses.txt")

# =================== PHASE 1: DATA PROCESSING PIPELINE ===================
print("Loading and processing data...")

campaign_response = pd.read_csv(CAMPAIGN_RESPONSE_PATH)
campaign_details  = pd.read_csv(CAMPAIGN_DETAILS_PATH)
region_data       = pd.read_csv(MASTER_LOOKUP_PATH)
transaction_data  = pd.read_csv(TRANSACTION_DATA_PATH)

# Build master dataset (exactly as in your notebook)
master1 = pd.merge(campaign_response, campaign_details, on="ChannelPartnerID", how="left")
master2 = pd.merge(master1, region_data, on="ChannelPartnerID", how="left")

total_sales_2021 = transaction_data[transaction_data["Year"] == 2021].groupby("ChannelPartnerID")["Sales"].sum().rename("total_sales_2021")
master3 = pd.merge(master2, total_sales_2021, on="ChannelPartnerID", how="left")

total_sales_2022 = transaction_data[transaction_data["Year"] == 2022].groupby("ChannelPartnerID")["Sales"].sum().rename("total_sales_2022")
master4 = pd.merge(master3, total_sales_2022, on="ChannelPartnerID", how="left")

brand_B1_sales_2022 = transaction_data[(transaction_data["Year"] == 2022) & (transaction_data["Brand"] == "B1")].groupby("ChannelPartnerID")["Sales"].sum().rename("brand_B1_sales_2022")
master5 = pd.merge(master4, brand_B1_sales_2022, on="ChannelPartnerID", how="left")

buying_frequency_2022 = transaction_data[transaction_data["Year"] == 2022].groupby("ChannelPartnerID")["Month"].nunique().rename("buying_frequency_2022")
master6 = pd.merge(master5, buying_frequency_2022, on="ChannelPartnerID", how="left")

brand_engagement_2022 = transaction_data[transaction_data["Year"] == 2022].groupby("ChannelPartnerID")["Brand"].nunique().rename("brand_engagement_2022")
master7 = pd.merge(master6, brand_engagement_2022, on="ChannelPartnerID", how="left")

buying_frequency_B1_2022 = transaction_data[(transaction_data["Year"] == 2022) & (transaction_data["Brand"] == "B1")].groupby("ChannelPartnerID")["Month"].nunique().rename("buying_frequency_B1_2022")
master8 = pd.merge(master7, buying_frequency_B1_2022, on="ChannelPartnerID", how="left")

master8["brand_B1_contribution_2022"] = np.where(
    master8["total_sales_2022"] > 0,
    master8["brand_B1_sales_2022"] / master8["total_sales_2022"], 0
)

transaction_data['date'] = pd.to_datetime(transaction_data['Year'].astype(str) + '-' + transaction_data['Month'].astype(str), format='%Y-%m')
transaction_data['Quarter'] = transaction_data['date'].dt.quarter

active_last_quarter = transaction_data[(transaction_data['Year'] == 2022) & (transaction_data['Quarter'] == 4)].groupby('ChannelPartnerID')['Sales'].sum()
active_last_quarter = active_last_quarter.apply(lambda x: 'Yes' if x > 0 else 'No').reset_index().rename(columns={'Sales': 'active_last_quarter'})
master9 = pd.merge(master8, active_last_quarter, on='ChannelPartnerID', how='left')
master9['active_last_quarter'] = master9['active_last_quarter'].fillna('No')

active_last_quarter_B1 = transaction_data[(transaction_data['Year'] == 2022) & (transaction_data['Quarter'] == 4) & (transaction_data['Brand'] == 'B1')].groupby('ChannelPartnerID')['Sales'].sum()
active_last_quarter_B1 = active_last_quarter_B1.apply(lambda x: 'Yes' if x > 0 else 'No').reset_index().rename(columns={'Sales': 'active_last_quarter_B1'})
final_master = pd.merge(master9, active_last_quarter_B1, on='ChannelPartnerID', how='left')

sales_cols = ["total_sales_2021", "total_sales_2022", "brand_B1_sales_2022",
              "brand_B1_contribution_2022", "buying_frequency_2022",
              "brand_engagement_2022", "buying_frequency_B1_2022"]
final_master[sales_cols] = final_master[sales_cols].fillna(0)
final_master['active_last_quarter_B1'] = final_master['active_last_quarter_B1'].fillna('No')

df = final_master.copy()
print(f"✓ Data processed: {len(df)} rows, {len(df.columns)} columns")

# =================== KPIs ===================
kpi_1 = df["ChannelPartnerID"].nunique()
kpi_2 = df["total_sales_2021"].sum()
kpi_3 = df["total_sales_2022"].sum()

# =================== MODEL DATA PREPARATION ===================
feature_cols_raw = [
    'loyalty', 'nps', 'n_yrs', 'email', 'sms', 'call',
    'brand_B1_contribution_2022', 'n_comp', 'portal', 'Region',
    'total_sales_2021', 'total_sales_2022', 'brand_B1_sales_2022',
    'buying_frequency_2022', 'brand_engagement_2022',
    'buying_frequency_B1_2022', 'active_last_quarter', 'active_last_quarter_B1'
]
target = 'response'

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

processed_train_df = train_df[feature_cols_raw].copy()
processed_test_df  = test_df[feature_cols_raw].copy()

for col in ['active_last_quarter', 'active_last_quarter_B1']:
    processed_train_df[col] = processed_train_df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    processed_test_df[col]  = processed_test_df[col].apply(lambda x: 1 if x == 'Yes' else 0)

processed_train_df = pd.get_dummies(processed_train_df, columns=['Region'], prefix='Region')
processed_test_df  = pd.get_dummies(processed_test_df,  columns=['Region'], prefix='Region')

for col in set(processed_train_df.columns) - set(processed_test_df.columns):
    processed_test_df[col] = 0
for col in set(processed_test_df.columns) - set(processed_train_df.columns):
    processed_train_df[col] = 0

processed_test_df = processed_test_df[processed_train_df.columns]

X_train = processed_train_df
y_train = train_df[target]
X_test  = processed_test_df
y_test  = test_df[target]

# =================== TRAIN MODELS ===================
print("Training models...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

blr_model = LogisticRegression(max_iter=1000, random_state=42)
blr_model.fit(X_train_scaled, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=20)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=150, random_state=42, max_features='sqrt',
                                   max_depth=8, min_samples_split=15, oob_score=True)
rf_model.fit(X_train, y_train)

models = {"blr": blr_model, "nb": nb_model, "dt": dt_model, "rf": rf_model}
print("✓ All models trained")

# =================== SENTIMENT ANALYSIS ===================
print("Running sentiment analysis...")

with open(REVIEWS_PATH, 'r', encoding='utf-8') as f:
    raw_data = f.read()

sia = SentimentIntensityAnalyzer()
reviews_list = [e.strip() for e in raw_data.split('\n') if e.strip()]
data_for_df = []

for entry in reviews_list:
    scores    = sia.polarity_scores(entry)
    compound  = scores['compound']
    sentiment = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')
    data_for_df.append({'Text': entry, 'Score': compound, 'Sentiment': sentiment})

sentiment_df    = pd.DataFrame(data_for_df)
sentiment_counts = sentiment_df['Sentiment'].value_counts()

# Word Cloud
stop_words = set(stopwords.words('english'))
stop_words.update(["coffee", "product", "tastes", "one", "try", "make", "taste"])
clean_text = re.sub(r'\[.*?\]\s*', '', raw_data)

wc = WordCloud(width=1000, height=500, background_color='white',
               stopwords=stop_words, colormap='viridis',
               max_words=100, relative_scaling=0.5).generate(clean_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Customer Feedback Word Cloud", fontsize=18, fontweight='bold', color='#08306B')
plt.tight_layout(pad=2)
wc_buf = BytesIO()
plt.savefig(wc_buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
wc_buf.seek(0)
encoded_wc = base64.b64encode(wc_buf.read()).decode()

# Sentiment Pie
ordered_sentiments = ['Positive', 'Negative', 'Neutral']
ordered_labels  = [s for s in ordered_sentiments if sentiment_counts.get(s, 0) > 0]
ordered_values  = [sentiment_counts.get(s, 0) for s in ordered_labels]
ordered_colors  = ['#4ECDC4', '#FF6B6B', '#FFE66D'][:len(ordered_labels)]

plt.figure(figsize=(8, 8))
plt.pie(ordered_values, labels=ordered_labels, autopct='%1.1f%%',
        startangle=90, colors=ordered_colors, explode=[0.05]*len(ordered_labels),
        shadow=True, textprops={'fontsize': 14, 'fontweight': 'bold'})
plt.title('Sentiment Distribution of Customer Reviews', fontsize=18, fontweight='bold', color='#08306B')
plt.axis('equal')
plt.tight_layout(pad=2)
pie_buf = BytesIO()
plt.savefig(pie_buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
pie_buf.seek(0)
encoded_pie = base64.b64encode(pie_buf.read()).decode()

print("✓ Sentiment analysis done")

# =================== DASH APP ===================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Needed for Render deployment

kpi_style = {
    "width": "32%", "padding": "15px",
    "border": "2px solid #08306B", "borderRadius": "12px",
    "textAlign": "center", "backgroundColor": "white",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
}

# =================== LAYOUT ===================
app.layout = html.Div([
    html.H1("360° FMCG Marketing Intelligence Dashboard", style={
        "textAlign": "center", "backgroundColor": "#08306B",
        "color": "white", "padding": "20px", "margin": 0,
        "fontSize": "28px", "fontWeight": "bold"
    }),
    html.Div([
        html.Div([
            dcc.Tabs(id="tabs", value="eda", vertical=True, children=[
                dcc.Tab(label="📊 EDA",       value="eda",       style={"padding": "40px 15px", "fontSize": "16px"}),
                dcc.Tab(label="🤖 Models",    value="model",     style={"padding": "40px 15px", "fontSize": "16px"}),
                dcc.Tab(label="💬 Sentiment", value="sentiment", style={"padding": "40px 15px", "fontSize": "16px"})
            ])
        ], style={"width": "120px", "backgroundColor": "#E6F0FA", "minHeight": "calc(100vh - 70px)"}),

        html.Div([html.Div(id="content-area")],
                 style={"flex": 1, "padding": "20px", "backgroundColor": "#f8f9fa", "overflowY": "auto"})
    ], style={"display": "flex"})
])

# =================== TAB ROUTING ===================
@app.callback(Output("content-area", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "eda":
        return html.Div([
            html.Div([
                html.Div([html.H4("Channel Partners"), html.H2(f"{kpi_1:,}")], style=kpi_style),
                html.Div([html.H4("Total Sales 2021"),  html.H2(f"₹{kpi_2:,.0f}")], style=kpi_style),
                html.Div([html.H4("Total Sales 2022"),  html.H2(f"₹{kpi_3:,.0f}")], style=kpi_style),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),

            html.Div([
                html.Label("📅 Select Year:", style={"fontWeight": "bold", "fontSize": "15px", "marginRight": "10px"}),
                dcc.Dropdown(id="year_dd",
                             options=[{"label": "2021", "value": "2021"}, {"label": "2022", "value": "2022"}],
                             value="2022", style={"width": "200px"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px",
                      "padding": "15px", "backgroundColor": "#E6F0FA", "borderRadius": "8px"}),

            html.Div([
                dcc.Graph(id="total_sales_box",    style={"width": "33.33%"}),
                dcc.Graph(id="sales_distribution", style={"width": "33.33%"}),
                dcc.Graph(id="regional_sales",     style={"width": "33.33%"}),
            ], style={"display": "flex"}),
            html.Div([
                dcc.Graph(id="buying_frequency_bar", style={"width": "33.33%"}),
                dcc.Graph(id="nps_box",              style={"width": "33.33%"}),
                dcc.Graph(id="complaints_box",       style={"width": "33.33%"}),
            ], style={"display": "flex"})
        ])

    if tab == "model":
        return html.Div([
            html.Div([
                html.Div([html.H4("Channel Partners"), html.H2(f"{kpi_1:,}")], style=kpi_style),
                html.Div([html.H4("Total Sales 2021"),  html.H2(f"₹{kpi_2:,.0f}")], style=kpi_style),
                html.Div([html.H4("Total Sales 2022"),  html.H2(f"₹{kpi_3:,.0f}")], style=kpi_style),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),

            html.Div([
                html.Label("🤖 Select Model:", style={"fontWeight": "bold", "marginRight": "10px", "fontSize": "15px"}),
                dcc.Dropdown(id="model_dd",
                             options=[
                                 {"label": "Logistic Regression", "value": "blr"},
                                 {"label": "Naive Bayes",         "value": "nb"},
                                 {"label": "Decision Tree",       "value": "dt"},
                                 {"label": "Random Forest",       "value": "rf"},
                             ],
                             value=None, placeholder="Select a model...",
                             style={"width": "300px"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px",
                      "padding": "15px", "backgroundColor": "#E6F0FA", "borderRadius": "8px"}),

            html.H4(id="model_header", style={"color": "#08306B", "marginBottom": "10px"}),
            html.Div([
                dcc.Graph(id="roc",  style={"width": "33%"}),
                dcc.Graph(id="cm",   style={"width": "33%"}),
                html.Div(id="third_chart", style={"width": "33%"}),
            ], style={"display": "flex"}),
            html.Div(id="dt_viz_container", style={"marginTop": "30px"})
        ])

    # Sentiment tab
    return html.Div([
        html.H3("Customer Sentiment Analysis",
                style={"textAlign": "center", "color": "#08306B", "marginBottom": "30px",
                       "fontSize": "24px", "fontWeight": "bold"}),
        html.Div([
            html.Div([html.H5("Total Reviews"),   html.H2(f"{len(sentiment_df)}")],
                     style={**kpi_style, "backgroundColor": "#f0f8ff"}),
            html.Div([html.H5("Positive Reviews"), html.H2(f"{sentiment_counts.get('Positive', 0)}",
                                                           style={"color": "#4ECDC4"})],
                     style={**kpi_style, "backgroundColor": "#e0f7f4"}),
            html.Div([html.H5("Negative Reviews"), html.H2(f"{sentiment_counts.get('Negative', 0)}",
                                                           style={"color": "#FF6B6B"})],
                     style={**kpi_style, "backgroundColor": "#ffe6e6"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "30px"}),

        html.Div([
            html.Div([
                html.H4("Word Cloud", style={"textAlign": "center", "color": "#08306B", "marginBottom": "10px"}),
                html.Img(src="data:image/png;base64," + encoded_wc,
                         style={"width": "100%", "borderRadius": "8px",
                                "border": "2px solid #08306B", "padding": "10px"})
            ], style={"width": "50%", "padding": "0 10px"}),
            html.Div([
                html.H4("Sentiment Distribution", style={"textAlign": "center", "color": "#08306B", "marginBottom": "10px"}),
                html.Img(src="data:image/png;base64," + encoded_pie,
                         style={"width": "100%", "borderRadius": "8px",
                                "border": "2px solid #08306B", "padding": "10px"})
            ], style={"width": "50%", "padding": "0 10px"}),
        ], style={"display": "flex"})
    ])

# =================== EDA CALLBACK ===================
@app.callback(
    [Output("total_sales_box", "figure"), Output("sales_distribution", "figure"),
     Output("regional_sales",  "figure"), Output("buying_frequency_bar", "figure"),
     Output("nps_box", "figure"),         Output("complaints_box", "figure")],
    Input("year_dd", "value")
)
def update_eda(year):
    sales_col = f"total_sales_{year}"

    fig1 = px.box(df, x="response", y=sales_col,
                  title=f"Total Sales {year} by Response",
                  color="response", color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"})
    fig1.update_layout(showlegend=False)

    df_h = df.copy(); df_h["response"] = df_h["response"].astype(str)
    fig2 = px.histogram(df_h, x=sales_col, color="response",
                        title=f"Sales Distribution {year}", nbins=30,
                        barmode="overlay", opacity=0.7,
                        color_discrete_map={"0": "#FF6B6B", "1": "#4ECDC4"})

    region_sales = df.groupby("Region")[sales_col].sum().reset_index()
    fig3 = px.bar(region_sales, x="Region", y=sales_col,
                  title=f"Total Sales by Region ({year})", color="Region")
    fig3.update_layout(showlegend=False)

    freq_data = df.groupby("response")["buying_frequency_2022"].mean().reset_index()
    fig4 = px.bar(freq_data, x="response", y="buying_frequency_2022",
                  title="Mean Buying Frequency 2022 by Response",
                  color="response", color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"})
    fig4.update_layout(showlegend=False)

    fig5 = px.box(df, x="response", y="nps",
                  title="NPS by Response",
                  color="response", color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"})
    fig5.update_layout(showlegend=False)

    fig6 = px.box(df, x="response", y="n_comp",
                  title="Number of Complaints by Response",
                  color="response", color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"})
    fig6.update_layout(showlegend=False)

    return fig1, fig2, fig3, fig4, fig5, fig6

# =================== MODEL CALLBACK ===================
@app.callback(
    [Output("roc", "figure"), Output("model_header", "children"),
     Output("cm", "figure"),  Output("third_chart", "children"),
     Output("dt_viz_container", "children")],
    Input("model_dd", "value")
)
def update_model(model_key):
    empty = go.Figure()
    empty.update_layout(title="Select a model to view results",
                        xaxis={"visible": False}, yaxis={"visible": False})
    if not model_key:
        return empty, "👆 Select a model above", empty, html.Div(), html.Div()

    model_names = {"blr": "Logistic Regression", "nb": "Naive Bayes",
                   "dt": "Decision Tree",         "rf": "Random Forest"}
    model = models[model_key]

    y_prob = model.predict_proba(X_test_scaled if model_key == "blr" else X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    cm          = confusion_matrix(y_test, y_pred)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                  name=f"AUC = {roc_auc:.3f}",
                                  line=dict(color='#08306B', width=2)))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  line=dict(dash='dash', color='gray'), name="Random"))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")

    cm_fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['No Response', 'Response'], y=['No Response', 'Response'],
                        color_continuous_scale='Blues')

    pred_df = pd.DataFrame({
        'Sample': range(1, 6),
        'Actual': y_test.iloc[:5].values,
        'Probability': y_prob[:5].round(4),
        'Predicted': y_pred[:5]
    })
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=['<b>Sample</b>', '<b>Actual</b>', '<b>Probability</b>', '<b>Predicted</b>'],
                    fill_color='#08306B', font=dict(color='white', size=13), align='center', height=35),
        cells=dict(values=[pred_df['Sample'], pred_df['Actual'],
                            pred_df['Probability'], pred_df['Predicted']],
                   fill_color='#E6F0FA', align='center', font=dict(size=12), height=30)
    )])
    table_fig.update_layout(title="Sample Predictions (first 5)", height=350,
                             margin=dict(t=50, b=10, l=10, r=10))

    third_chart = dcc.Graph(figure=table_fig)
    header      = f"Model: {model_names[model_key].upper()}  |  AUC Score: {roc_auc:.4f}"

    additional_viz = html.Div()

    if model_key == "dt":
        plt.figure(figsize=(24, 12))
        plot_tree(dt_model, feature_names=X_train.columns,
                  class_names=['No Response', 'Response'],
                  filled=True, rounded=True, fontsize=9)
        plt.title('Decision Tree Visualization (max_depth=5)', fontsize=16)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        additional_viz = html.Div([
            html.H3("Decision Tree Visualization", style={"textAlign": "center", "color": "#08306B"}),
            html.Img(src=f"data:image/png;base64,{img_b64}", style={"width": "100%"})
        ])

    elif model_key == "rf":
        fi = pd.DataFrame({'Feature': X_train.columns,
                           'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi.head(15), x='Importance', y='Feature', palette='viridis')
        plt.title('Top 15 Feature Importances — Random Forest', fontsize=16)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        additional_viz = html.Div([
            html.H3("Random Forest — Feature Importance", style={"textAlign": "center", "color": "#08306B"}),
            html.Img(src=f"data:image/png;base64,{img_b64}",
                     style={"width": "80%", "display": "block", "margin": "0 auto"})
        ])

    return roc_fig, header, cm_fig, third_chart, additional_viz

# =================== RUN ===================
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8050)))
