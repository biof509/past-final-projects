import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Slides and resources can be found at: https://github.com/cianadeveau/BIOF509-Final

# Download Basic School Stats and Basic Opponent Stats from https://www.sports-reference.com/cbb/seasons/2019-school-stats.html for the current season as an excel file

# One Function to import and clean


def clean_excel(file1:str, file2:str):
    ''' Imports the test data
        Cleans and preprocesses the dataframes
        Arguments:
            file1 = name of excel file as a string
            file2 = name of excel file as a string'''
    df = pd.read_excel(file1, header=1)
    df2 = pd.read_excel(file2, header=1)

    # Select Tournament teams
    df = df[df['School'].str.contains('NCAA')]
    df['School'] = df['School'].str.replace('NCAA','')
    df2 = df2[df2['School'].str.contains('NCAA')]
    df2['School'] =df2['School'].str.replace('NCAA','')

    # Remove unwanted columns
    labels = ['Rk', 'G', 'W', 'L', 'SRS', 'W.1', 'L.1', 'W.2',
              'L.2', 'W.3', 'L.3', 'MP', 'FG', 'FGA', '3P', '3PA',
              'FT', 'FTA', 'Unnamed: 16', 'ORB', 'STL']
    df = df.drop(labels, axis=1)
    labels = ['Rk', 'School', 'G', 'W', 'L', 'W-L%', 'SOS', 'SRS', 'W.1', 'L.1',
                'W.2', 'L.2', 'W.3', 'L.3', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT',
                'FTA', 'Unnamed: 16', 'Tm.', 'Opp.', 'ORB', 'STL']
    df2 = df2.drop(labels, axis=1)

    # Change column names
    dict = {'FG%': 'FG%O', '3P%': '3P%O', 'FT%': 'FT%O', 'TRB': 'TRBO', 'AST': 'ASTO',
            'BLK': 'BLKO', 'TOV': 'TOVO', 'PF': 'PFO'}
    df2 = df2.rename(columns=dict)

    # Join the dataframes
    df_test = df.join(df2, how='left')

    return df_test


# Second Function to apply the model


def predict_rounds(df_test, output, model_type, name=None, train_file='NCAA.xlsx'):
    ''' Creates and runs model
        Returns either the probabilities or the prediction
        Arguments:
            df_test = the file made from clean_excel
            train_file = default is NCAA.xlsx - file pre-cleaned for this model found here:
                         https://github.com/cianadeveau/BIOF509-Final/blob/master/NCAA.xlsx
            output = what you want returned, 'prob' = probabilities, 'pred' = prediction
            model_type = either SVC ('svc') or RandomForestClassifier ('rfc')
            name = option to return only one name from the test file'''
    # read in the train_file and set X, y
    train = pd.read_excel(train_file)
    X = train.iloc[:, 1:-1]
    y = train.Label

    # set X_test and names variable
    names = df_test
    names['School'] = df_test['School'].str.replace('\xa0', '')
    name_index = names[['School']].reset_index(drop=True)
    if name in names.School.unique():
        row = pd.DataFrame(names[names.School == name])
        X_test = row.iloc[:, 1:]
    elif name not in names.School.unique() and name != None:
        X_test = df_test.iloc[:, 1:]
        print('Sorry that school did not make the tournament. Here are model results for the schools that did.')
    elif name == None:
        X_test = df_test.iloc[:, 1:]

    # Pipeline with scaler and RFC
    if model_type == 'rfc':
        steps = [('scaler', StandardScaler()),
                 ('RFC', RandomForestClassifier(n_estimators=120, random_state=5))]
        pipeline = Pipeline(steps)
        model = pipeline.fit(X, y)

    if model_type == 'svc':
        steps = [('scaler', StandardScaler()),
                 ('SVC', SVC(gamma='auto', kernel='linear', probability=True))]
        pipeline = Pipeline(steps)
        model = pipeline.fit(X, y)

    # Return probabilities or Return labels only
    if output == 'prob':
        probs = pd.DataFrame(model.predict_proba(X_test))
        probs.columns = model.classes_
        if name != None and name in names.School.unique():
            statement = '{name}: '.format(name=name) + str(probs)
            return statement
        else:
            probs = name_index.join(probs, how='left')
            return probs
    if output == 'pred':
        preds = model.predict(X_test)
        if name != None and name in names.School.unique():
            statement = '{name}: '.format(name=name) + str(preds)
            return statement
        else:
            preds = pd.DataFrame(preds)
            preds = name_index.join(preds, how='left')
            preds.columns = ['School', 'Round']
            return preds


test = clean_excel('2019Tournament.xlsx', '2019Tournament_opp.xlsx')
probs = predict_rounds(df_test = test, output = 'prob', model_type='svc')
print(probs)
