
# Spam-Detection
This project is aimed at detecting spam messages using machine learning algorithms. The project uses a dataset of messages that have been previously labeled as either spam or not spam (ham), and uses this data to train various machine learning models to predict whether new, unseen messages are spam or not.

# Prerequisites
<h3>To run this project, you will need the following:<br></h3>

Python 3.x<br>
Jupyter Notebook<br>
scikit-learn library<br>
pandas library<br>
numpy library<br>


let’s have a look at the first five rows of this dataset:</h3>
data.head()<br>

![Screenshot (24)](https://user-images.githubusercontent.com/110754364/235336061-87192c7c-b060-478a-bed3-f3b350ba85b5.png)

<h3>let's check for the size of the dataset</h3>
data.shape<br>
(150, 5)<br>

<h3>From this dataset, class and message are the only features we need to train a machine learning model for spam detection, so let’s select these two columns as the new dataset:</h3>
data = data[["v1", "v2"]]<br>

<h3>Now let’s split this dataset into training and test sets and train the model to detect spam messages:</h3>
x = np.array(data["message"])<br>
y = np.array(data["class"]),br>
cv = CountVectorizer()<br>
X = cv.fit_transform(x) # Fit the Data<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)<br>

clf = MultinomialNB()<br>
clf.fit(X_train,y_train)<br>

<h3>Now let’s test this model by taking a user input as a message to detect whether it is spam or not:</h3>
sample = input('Enter a message:')<br>
data = cv.transform([sample]).toarray()<br>
print(clf.predict(data))<br>

Enter a message:You won $40 cash price
['spam']

