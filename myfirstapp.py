import streamlit as st

import numpy as np
import base64
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px





# Page photo header
image= Image.open('customer.jpg')
st.image(image, use_column_width=True)

# Page title
st.title("Customer Personality Analysis")

st.markdown("""
Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers
and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.

This app performs clustering to summarize customer segments.
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Kaggle.com](https://www.kaggle.com/imakash3011/customer-personality-analysis)
***
""")

# sidebar feature
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2020))))

# display data 
st.write(" ### Customer Personality dataset:")

path = 'marketing_campaign.csv' 
df = pd.read_csv(path)
st.write('*Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.*')

df.head(10)
st.dataframe(df)

# Download customers stats data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="marketing_campaign.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df), unsafe_allow_html=True)

# heatmap button
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df.to_csv('/content/drive/MyDrive/marketing_campaign.csv',index=False)
    #df = pd.read_csv('marketing_campaign.csv')

    cor = df.corr()
    plt.figure(figsize = (27,26))
    sns.heatmap(cor, annot = True, cmap = 'coolwarm')
    plt.show()
    st.pyplot()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)


st.write("""
***
""")

# side bar 1
option = st.sidebar.selectbox(
    'Select a data analysis',
     ['Line Chart','Income Distribution','Age Range'])

# A bar graph showing the cumulative total by educational background

if option=='Line Chart':
    chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

    
    
elif option=='Income Distribution':

    response = df.query('Response == 1')
    non_response = df.query('Response == 0')
    fig = go.Figure()
    fig.add_trace(go.Violin(x=response['Income'], line_color='lightseagreen', name='Response', y0=0))
    fig.add_trace(go.Violin(x=non_response['Income'], line_color='red', name= 'Don\'t response', y0=0))

    fig.update_traces(orientation='h', side='positive', meanline_visible=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

    fig.update_layout(title='<b>Income distribution (Response Vs Non-response)<b>',
                  xaxis_title='Income',
                  titlefont={'size': 18},
                  width=600,
                  height=400,
                  template="plotly_dark",
                  showlegend=True,
                  paper_bgcolor="lightgray",
                  plot_bgcolor='lightgray', 
                  font=dict(
                      color ='black',
                      )
                  )
    fig.update_layout(xaxis_range=[0,150000])
    fig.show()
    st.write("### The Distribution Analysis")
    st.write("The distribution below shows that income has little impact on response, which is good news because if it did, it would mean that the products are expensive because many customers would not buy because of the high prices, but this does not appear to be the case, so we must look for another explanation.")
    st.plotly_chart(fig, use_container_width=True)


elif option=='Age Range':
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['Age'] = df['Dt_Customer'].dt.year - df['Year_Birth']
    df['Age_Range'] = 'Child'
    df.loc[df['Age'] >= 13, 'Age_Range'] = 'Teens'
    df.loc[df['Age'] >= 21, 'Age_Range'] = "Youth"
    df.loc[df['Age'] >= 36, 'Age_Range'] = "Old"

    fig = go.Figure()
    fig.add_trace(
      go.Pie(
        labels=df['Age_Range'],
        values=None,
        hole=.4,
        title='Age Range',
        titlefont={'color':None, 'size': 18},
        )
      )
    fig.update_traces(
      hoverinfo='label+value',
      textinfo='label+percent',
      textfont_size=12,
      marker=dict(
          colors=['lightgrey', 'lightseagreen', 'lightblue'],
          line=dict(color='#000000',
                    width=2)
          )
      )
    fig.show()

    st.write('### Customer\'s Age Range' )
    st.plotly_chart(fig)
    st.write('The target market is those aged 36 and up, thus we need to sell things that cater to this demographic.')


st.write("""
***
""")




