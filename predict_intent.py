import pandas as pd
import random
import spacy
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def select_answer(intent):
    global df
    i = random.randint(0, len(df[intent]))
    answer = df.at[i, intent]
    print(answer + " (" + intent +")")

def predict_intent():
    predicted_intents = []
    interpreter = Interpreter.load('./models/nlu/default/aes_bot_v0')
    # print(interpreter.parse(text))
    # return interpreter.parse(text)
    sentence = ""
    print ("Começa chat")
    while sentence is not "quit":
        sentence = input()
        sentence = str(sentence)
        # print(sentence)
        intent = interpreter.parse(sentence)
        intent = intent["intent"]
        intent = intent["name"]
        select_answer(intent)

        # predicted_intents.append(intent)
        # print(predicted_intents[0] + ": "+ select_answer(predicted_intents[0]))
        # predicted_intents = []

answers = 'answers.csv'
df = pd.read_csv(answers, sep = '\t')
predict_intent()