import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_title(df):
    title = pd.DataFrame()

    title['Title'] = df['Name'].map(lambda name: name.split(',')[
                                    1].split('.')[0].strip())

    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    title['Title'] = title.Title.map(title_dictionary)
    title = pd.get_dummies(title.Title)

    return title


def extract_features(data):
    embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')
    tittle = get_title(data)
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    feature_columns = [
        # 'Fare',
        # 'Age',
        # 'SibSp',
        # 'Parch',
        # 'Ticket'
    ]

    feature_columns += list(embarked.columns)
    feature_columns += list(tittle.columns)

    data = pd.concat([data, embarked, tittle], axis=1)
    return data[feature_columns]


train = pd.read_csv('data/train.csv')
target = train['Survived']

train_features = extract_features(train)


train_data, valid_data, train_target, valid_target = train_test_split(
    train_features, target, train_size=.7, test_size=.3)
model = RandomForestClassifier(n_estimators=100)
model.fit(train_data, train_target)

print(model.score(train_data, train_target),
      model.score(valid_data, valid_target))
