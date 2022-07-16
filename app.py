import streamlit as st
import pandas as pd
import PROCESS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
user_menu = st.sidebar.radio(
    'Select an Option',
    ('IPL TALLY','OVERALL ANALYSIS','TEAM-WISE ANALYSIS','BOLING-WISE ANALYSIS','BATTING-WISE ANALYSIS')
)
if user_menu == 'IPL TALLY':

    st.sidebar.header("Medal Tally")
    years,country = PROCESS.country_year_list()
    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)
    medal_tally = PROCESS.fetch_model_tally(selected_year, selected_country)
    st.table(medal_tally)
if user_menu == 'OVERALL ANALYSIS':
    dATA = PROCESS.preprocess()
    st.dataframe(dATA)
if user_menu == 'BATTING-WISE ANALYSIS':
    st.title("Most Runs:-")
    boling_file= pd.read_excel("best   bolling.xlsx" ,1)
    boling_file=boling_file.set_index(['Rank'])
    boling_file.index.name = 'Rank'
    st.dataframe(boling_file)
    st.title("Most Sixes:-")
    boling_file = pd.read_excel("best   bolling.xlsx", sheet_name='Most Sixes')
    boling_file = boling_file.set_index(['Rank'])
    st.dataframe(boling_file)
    st.title("Most Fours:-")
    boling_file = pd.read_excel("best   bolling.xlsx",sheet_name='Most Fours')
    boling_file = boling_file.set_index(['Rank'])
    st.dataframe(boling_file)
    import plotly.figure_factory as ff
    df = pd.read_csv("Players18.csv")
    df = pd.DataFrame(df)
    st.dataframe(df)
    df = df.iloc[:, 1:]
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=0)
    scaler = MinMaxScaler()
    # fit the scaler to the train set, it will learn the parameters
    scaler.fit(X_train)
    # transform train and test sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    x1 = X_train_scaled['age']
    x2 = X_train_scaled['run']
    fig = ff.create_distplot([x1, x2], ['AGE','RUNS'], curve_type='normal', show_hist=False, show_rug=False)
    st.plotly_chart(fig)
if user_menu == 'OVERALL ANALYSIS':
    st.title("Teams over the years")
    df1 = pd.read_csv("matches.csv")
    df2 = df1.drop(['umpire3'], axis=1)
    df2 = df2.dropna()
    df2 = df2.reset_index(drop=True)
    group_by = df2.groupby('Year')['team1'].nunique().reset_index()
    group_by.rename(columns={'team1': 'No Of Teams '}, inplace=True)
    fig = px.line(group_by, x="Year", y="No Of Teams ")
    st.plotly_chart(fig)
    st.title("City over the years")
    df1 = pd.read_csv("matches.csv")
    df2 = df1.drop(['umpire3'], axis=1)
    df2 = df2.dropna()
    df2 = df2.reset_index(drop=True)
    group_by = df2.groupby('Year')['city'].nunique().reset_index()
    group_by.rename(columns={'city': 'No Of City'}, inplace=True)
    fig = px.line(group_by, x="Year", y='No Of City')
    st.plotly_chart(fig)
    st.title("Winning and Losing Analysis by Winning Toss ")
    dff = pd.DataFrame({'x': ['win_match', 'Loose_match'], 'y': [388, 355]})
    fig = px.pie(dff, values='y', names='x', color_discrete_sequence=px.colors.sequential.RdBu, )
    fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    st.title("Winning and Losing Analysis by chossing bat or field ")
    df23 = pd.DataFrame({'x': ['choose_field', 'choose bat'],'y':[256,132]})
    fig = px.pie(df23, values='y', names='x', color_discrete_sequence=px.colors.sequential.RdBu, )
    fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    st.title("PLAY THE TEAM OVER THE YEAR ")
    data = pd.read_excel("heatmap.xlsx",sheet_name='Sheet3')
    data_set = pd.DataFrame(data.iloc[0:15, 0:15])
    fig, ax = plt.subplots(figsize=(30, 30))
    ax = sn.heatmap(data_set, annot=True)
    yticks_labels = ['RR', 'CSK', 'MI', 'KKR', 'RCB', 'De', 'DC', 'PBKS', 'GT', 'LSG', 'SRH', 'GL', 'KTK', 'PWI', 'RPS']
    plt.yticks(np.arange(15) + .5, labels=yticks_labels, fontsize=30)
    plt.xticks(fontsize=30)
    st.pyplot(fig)
if user_menu == 'TEAM-WISE ANALYSIS':
    list = ['Highest Totals', 'Lowest Totals',
            'Wicketkeeping-Most career catch',
            'Wicketkeeping-Most career stump',
            'Fielding-Most career catches',
            'AllRounder500 Runsand 20 wicket',
            'most-matches-of-player',
            'Most matches as captain',
            'Most matches won as a captain',
            'Most runs as captain',
            'Most man of the match awards']
    list2 = st.selectbox('Select a Sport', list)
    var1 = pd.read_excel("filter.xlsx", 'Highest Totals')
    var2 = pd.read_excel("filter.xlsx", 'Lowest Totals')
    var3 = pd.read_excel("filter.xlsx", 'Wicketkeeping-Most career catch')
    var4 = pd.read_excel("filter.xlsx", 'Wicketkeeping-Most career stump')
    var5 = pd.read_excel("filter.xlsx", 'Fielding-Most career catches')
    var6 = pd.read_excel("filter.xlsx", 'AllRounder500 Runsand 20 wicket')
    var7 = pd.read_excel("filter.xlsx", 'most-matches-of-player')
    var8 = pd.read_excel("filter.xlsx", 'Most matches as captain')
    var9 = pd.read_excel("filter.xlsx", 'Most runs as captain')
    var10 = pd.read_excel("filter.xlsx", 'Most man of the match awards')
    var11=pd.read_excel("filter.xlsx", 'Most matches won as a captain')
    if list2 == 'Most man of the match awards':
        st.dataframe(var10)
    elif list2 == 'Highest Totals':
        st.dataframe(var1)
    elif list2 == 'Lowest Totals':
        st.dataframe(var2)
    elif list2 == 'Wicketkeeping-Most career catch':
        st.dataframe(var3)
    elif list2 == 'Wicketkeeping-Most career stump':
        st.dataframe(var4)
    elif list2 == 'Fielding-Most career catches':
        st.dataframe(var5)
    elif list2 == 'AllRounder500 Runsand 20 wicket':
        st.dataframe(var6)
    elif list2 == 'most-matches-of-player':
        st.dataframe(var7)
    elif list2 == 'Most matches as captain':
        st.dataframe(var8)
    elif list2 == 'Most runs as captain':
        st.dataframe(var9)
    elif list2 == 'Most matches won as a captain':
        st.dataframe(var11)
if user_menu == 'BOLING-WISE ANALYSIS':
    import plotly.figure_factory as ff
    var12 = pd.read_excel("best_boweler.xlsx")
    df = pd.DataFrame(var12)
    st.dataframe(df)
    df = pd.DataFrame(df)
    df = df.iloc[:, 1:]
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=0)
    scaler = MinMaxScaler()
    # fit the scaler to the train set, it will learn the parameters
    scaler.fit(X_train)
    # transform train and test sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    x1 = X_train_scaled['age']
    x2 = X_train_scaled['WICKET']
    fig = ff.create_distplot([x1, x2], ['AGE', 'WICKET'], curve_type='normal', show_hist=False, show_rug=False)
    st.plotly_chart(fig)

