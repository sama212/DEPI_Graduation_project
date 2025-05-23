import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

data = pd.read_csv(r"C:\Users\motherboard\Desktop\data\cleaned_data.csv")


numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']


fig = make_subplots(rows=2, cols=3, subplot_titles=numerical_features)

for i, feature in enumerate(numerical_features, 1):
    fig.add_trace(go.Histogram(x=data[feature], nbinsx=30, name=feature), row=(i-1)//3+1, col=(i-1)%3+1)

fig.update_layout(title_text="Distribution of Numerical Features", showlegend=False)
fig.show()


fig = make_subplots(rows=2, cols=3, subplot_titles=numerical_features)

for i, feature in enumerate(numerical_features, 1):
    fig.add_trace(go.Box(x=data['num'], y=data[feature], name=feature), row=(i-1)//3+1, col=(i-1)%3+1)

fig.update_layout(title_text="Boxplots of Numerical Features vs Target (num)", showlegend=False)
fig.show()

num_counts = data['num'].value_counts()

fig1 = px.pie(
    values=num_counts.values,
    names=num_counts.index.map({0: 'No Heart Disease', 1: 'Heart Disease'}),  
    title='Distribution of Target Variable (num)',
    hole=0.3,
    color=num_counts.index.map({0: 'No Heart Disease', 1: 'Heart Disease'}),
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig1.update_layout(
    legend=dict(
        title='Target Variable',  
        orientation='h',  
        yanchor='bottom',  
        y=-0.2, 
        xanchor='center',  
        x=0.5, 
        font=dict(size=12)  
    )
)

fig1.update_traces(
    textposition='inside',
    textinfo='percent+label',
    pull=[0.1, 0]
)
fig1.show()

sex_counts = data['sex_Female'].value_counts()

fig2 = px.pie(
    values=sex_counts.values,
    names=sex_counts.index.map({0: 'Male', 1: 'Female'}),  
    title='Distribution of Gender (sex_Female)',
    hole=0.3,
    color=sex_counts.index.map({0: 'Male', 1: 'Female'}),
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig2.update_layout(
    legend=dict(
        title='Gender',  
        orientation='h', 
        yanchor='bottom',
        y=-0.2, 
        xanchor='center',  
        x=0.5,  
        font=dict(size=12)  
    )
)

fig2.update_traces(
    textposition='inside',
    textinfo='percent+label',
    pull=[0.1, 0]
)
fig2.show()