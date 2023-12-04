
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import quote
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv(r"C:\Users\kashinath konade\Desktop\CodSoft Intern for DS\creditcard.csv", encoding='unicode_escape')

# Save the data to an SQL database
user = 'root'
pw = 'kashinath@123'
db = 'Assignment_ml'
engine = create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))
df.to_sql('Movie_Rating'.lower(), con=engine, if_exists='replace', index=False)

# Read the data from the MySQL database

sql = 'SELECT * FROM Movie_Rating'
movie = pd.read_sql_query(sql, con=engine)


#Check the first few rows of the dataset
movie.head()

#Get the shape of the dataset
movie.shape

# Checking Null values
movie.isnull().sum()

# Separating the data for analysis

# 0 -> Normal Transaction
# 1 -> Fraudulent Transaction

legit = movie[movie.Class == 0]

fraud = movie[movie.Class == 1]

# Satastical measures of the data

print(legit.Amount.describe())

print(fraud.Amount.describe())

# Compare the data for both transactions

print(movie.groupby('Class').mean())

# Under-sampling
# Build a sample dataset containing a similar distribution of normal and fraudulent transactions

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Splitting the data into features and targets

X = new_dataset.drop(columns='Class', axis=1)

Y = new_dataset['Class']

# Split the data into Training and Testing Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#Model Training 

# Logistic Regression

model = LogisticRegression()

# Training the Logistic Regression Model With Training Data

model.fit(X_train, Y_train)

# Model Evaluation

# Accuracy Score

# Accuracy on Training data

# Checking Accuracy for Training data
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data: ', training_data_accuracy)

# Checking Accuracy for Test Data 

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data: ', test_data_accuracy)


# Confusion Matrix

conf_matrix = confusion_matrix(Y_test, X_test_prediction)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()