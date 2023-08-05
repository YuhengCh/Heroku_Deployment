import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
# load model
with open('model_iris.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')  # http://www.google.com/
def home():

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    print('Received input:', int_features)
    final_features = [np.array(int_features)]
    print('Final input:', final_features)
    prediction = model.predict(final_features)

    feature_names = ['setosa', 'versicolor', 'virginica']
    result = feature_names[int(prediction[0])]

    return render_template('index.html', prediction_text='Predicted Iris Species: {}'.format(result))


if __name__ == "__main__":
    app.run(port=5000, debug=True)
