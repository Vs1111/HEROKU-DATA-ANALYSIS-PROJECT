import pandas as pd
global df,df1,df2,df3
df = pd.read_csv("real ipl datset\points_table.csv")
df1 =pd.read_excel("real ipl datset\\homes_add.xlsx")
df.rename(columns={'name': 'Team_Name', 'matcheswon': 'Matches_Won'}, inplace=True)
df2 =pd.read_excel("real ipl datset\\Copy of Book1.xlsx")
df3 =pd.read_csv("real ipl datset\\matches.csv")

def preprocess():
    var = df.groupby(['Team_Name']).sum()[['matcheslost']]
    var1 = df.groupby(['Team_Name']).sum()[['matchpoints']]
    var2 = df.groupby(['Team_Name']).sum()[['Matches_Won']]
    var3 = pd.concat([var2, var1, var], axis=1)
    points = pd.merge(var3, df1, how="left", on=["Team_Name"])
    points = pd.merge(points, df2, how="left", on=["Team_Name"])
    points = points.sort_values('Matches_Won', ascending=False)
    return points
def country_year_list():

    years = df3['Year'].unique().tolist()
    years.sort()
    years.insert(0,'Overall')
    Team_name =df3['team1'].unique().tolist()
    Team_name.sort()
    Team_name.insert(0, 'Overall')
    return years,Team_name
df19 = df3[['Year','team1','team2','winner','player_of_match','venue','TEAM_NAME']]
df19 = df19.dropna()
df19.isnull().sum()
def fetch_model_tally(year, Team_name):

  if year =='Overall' and Team_name =='Overall':
     tempdf = df19
  if year =='Overall' and Team_name!='Overall':
      mask = df19['TEAM_NAME'].str.contains(Team_name, case=False, na=False)
      tempdf = df19[mask]
  if year !='Overall' and Team_name =='Overall':
       tempdf = df19[df19['Year'] == int(year)]
  if year !='Overall' and Team_name !='Overall':
        tempdf = df19[(df19['Year'] == int(year) ) & (df19['TEAM_NAME'].str.contains(Team_name, case=False, na=False))]
  return tempdf