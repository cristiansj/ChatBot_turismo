#importaciones
import tflearn
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
import nltk
from flask import Flask, render_template, request, session
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')

#Preprocesamiento de los datos de entrenamiento usando NTLK
words = []
classes = []
documents = []
ignore_words = ['?']
data_file = open('files/turismo.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        # add to words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# perform stemming and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print(words,"\n",classes,"\n",documents,"\n")

#Se crean las estructuras de datos para entrenar el chatbot

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    output.append(output_row)

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
output = np.array(output)

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

#Se crea la red neuronal usando TensorFlow y TFLearn

tf.compat.v1.reset_default_graph()

# build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

#Se entrena el chatbot con el conjunto de datos que ya hemos definido

# start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=30, show_metric=True)
model.save('model.tflearn')

#Se carga el modelo entrenado y se crean las funciones para procesar las entradas del usuario y generar las respuestas

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))

def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict([p])[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0][0]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Función para guardar los datos de la conversación en el dataframe y en un archivo Excel
df = pd.read_excel("files/log.xlsx")

def guardar_conversacion(usuario, pr, rp, tag):
    global df
    nueva_fila = {'Usuario': usuario, 'Pregunta': pr, 'Respuesta': rp, 'Tag': tag}
    df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
   # df = df.append({'Usuario': usuario, 'Pregunta': pr, 'Respuesta': rp, 'Tag': tag}, ignore_index=True)
    df.to_excel('files/log.xlsx',index=False)

app = Flask(__name__, static_folder='static')
app.secret_key = 'my_secret_key'

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('login.html')

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with open("files/usuarios.json", encoding='utf-8') as file:
            usuarios = json.load(file)

        cedula = request.form['cedula']
        if cedula in usuarios:
            nombre = "Bienvenido compadre, "
            nombre += usuarios[str(cedula)]['nombre']
            session['usuario_sesion'] = cedula
            return render_template('chatbot.html', username=nombre)
        else:
            return render_template('registro.html')

    return render_template('login.html')

@app.route('/registro.html', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        with open("files/usuarios.json", encoding='utf-8') as file:
            usuarios = json.load(file)

        cedula = request.form['cedula']
        if cedula in usuarios:
            nombre = "Bienvenido compadre, "
            nombre += usuarios[str(cedula)]['nombre']
            session['usuario_sesion'] = cedula
            return render_template('chatbot.html', username=nombre)
        else:
            nombre = request.form['nombre']
            edad = int(request.form['edad'])
            pais = request.form['pais']
            pais = pais.lower()
            ciudad = request.form['ciudad']
            ciudad = ciudad.lower()
            genero = request.form['genero']
            genero = genero.lower()
            usuarios[cedula] = {"nombre": nombre,"edad":edad,"pais":pais,"ciudad":ciudad, "genero": genero}
            with open("files/usuarios.json", "w", encoding='utf-8') as file:
                json.dump(usuarios, file)
            nombreA = "Bienvenido compadre, "
            nombreA += nombre
            session['usuario_sesion'] = cedula
            return render_template('chatbot.html', username=nombreA)

    return render_template('registro.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    # predict intent
    ints = predict_class(userText)
    if len(ints) == 0:
        return 'Ay, hijuepucha, no entendí, repitímela más despacio, ¿puede ser?'
    # get response
    res = getResponse(ints, intents)
    guardar_conversacion(session['usuario_sesion'], userText, res, ints[0][0])
    return res

@app.route('/baseDatos')
def base_datos():
    # Lee el archivo Excel
    df1 = pd.read_excel('files/log.xlsx')
    df1 = df1.rename(columns={'Usuario': 'cedula'})
    df1['cedula'] = df1['cedula'].astype(str)

    # Lee el archivo JSON
    with open('files/usuarios.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    df2 = pd.DataFrame.from_dict(data, orient='index', columns=['nombre', 'edad', 'pais', 'ciudad', 'genero'])
    df2.index.name = 'cedula'

    # Hacer el join utilizando la columna 'cedula'
    df_merged = pd.merge(df1, df2, left_on='cedula', right_index=True)

    # Guardar el DataFrame en un archivo Excel
    df_merged.to_excel('base_de_datos_turismo.xlsx', index=False)

    return 'se creo la base de datos'



if __name__ == '__main__':
    app.run(port=7000)
