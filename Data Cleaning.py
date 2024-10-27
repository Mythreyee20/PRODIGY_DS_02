import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)
print(titanic.head())

print("\nMissing values in each column:")
print(titanic.isnull().sum())

titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic.drop('Cabin', axis=1, inplace=True)
  
titanic.dropna(subset=['Embarked'], inplace=True)
print("\nData types:")
print(titanic.dtypes)
titanic['Survived'] = titanic['Survived'].astype('category')
titanic['Pclass'] = titanic['Pclass'].astype('category')
titanic['Sex'] = titanic['Sex'].astype('category')

plt.figure(figsize=(10, 5))
sns.histplot(titanic['Age'], bins=30, kde=True)
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=titanic, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=titanic, x='Sex', hue='Survived')
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='Survived', y='Age', data=titanic)
plt.title('Age Distribution by Survival')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
