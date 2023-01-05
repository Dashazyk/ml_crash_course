"""
----------------------------
Web-service with API
----------------------------
"""
from email import message
import os
import unicodedata


import pickle
import numpy as np
from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse

from src.utils import conf, logger, MessagesDB, ML

from catboost import CatBoostClassifier

db = MessagesDB(conf)
db.init_db()

def load_model(model_path):
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)

    logger.info("Loaded model from disk")
    return loaded_model

def load_vectorizer(vectorizer_path):
    loaded_vectorizer = pickle.load(open(vectorizer_path))

    logger.info("Loaded vectorizer from disk")
    return loaded_vectorizer


app = Flask(__name__)
api = Api(app)
model = load_model(conf.model_path)
vectorizer = load_vectorizer(conf.vectorizer_path)

@app.route('/messages/<string:identifier>')
def predict_label(identifier):
    msg = db.read_message(msg_id=int(identifier))

    # model predict single label
    pre_proc = ML.preprocessing(np.array( [msg] ))
    X_test_csr = vectorizer.transform(pre_proc)
    pred = model.predict(X_test_csr)
    predicted_label = pred[0]

    return render_template('page.html', id=identifier, txt=msg['txt'], label=predicted_label)

@app.route('/feed/')
def feed():
    limit = request.args.get('limit', 10)
    limit = int(limit)
    # rank all messages and predict
    msg_ids = db.get_messages_ids(limit)

    messages = []

    for id in msg_ids:
        msg = db.read_message(msg_id=int(id)) 
        # model predict single label
        pre_proc = ML.preprocessing(np.array( [msg] ))
        X_test_csr = vectorizer.transform(pre_proc)
        pred = model.predict(X_test_csr)
        predicted_label = pred[0]

        message = {}
        message['msg_id'] = id
        message['msg_txt'] = msg
        message['msg_pred'] = predicted_label

        messages.append(message)

    sorted_messages = sorted(messages, key=lambda d: d['msg_pred'], reverse = True) 

    return render_template('feed.html', recs=sorted_messages)

class Messages(Resource):
    def __init__(self):
        super(Messages, self).__init__()
        self.msg_ids = db.get_messages_ids()  # type: list

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('limit', type=int, default=10, location='args')
        args = parser.parse_args()
        try:
            resp = [int(i) for i in np.random.choice(self.msg_ids, size=args.limit, replace=False)]
        except ValueError as e:
            resp = 'Error: Cannot take a larger sample than %d' % len(self.msg_ids)
        return {'msg_ids': resp}


api.add_resource(Messages, '/messages')
logger.info('App initialized')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
