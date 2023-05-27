from flask import Flask, render_template, request
import json

app = Flask(__name__, static_folder='static')

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with open("files/usuarios.json") as file:
            usuarios = json.load(file)

        cedula = request.form['cedula']
        if cedula in usuarios:
            return render_template('chatbot.html')
        else:
            return render_template('registro.html')

    return render_template('login.html')

@app.route('/registro.html', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        with open("files/usuarios.json") as file:
            usuarios = json.load(file)

        cedula = request.form['cedula']
        if cedula in usuarios:
            return render_template('chatbot.html')
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
            with open("files/usuarios.json", "w") as file:
                json.dump(usuarios, file)
            return 'Registro completo!'

    return render_template('registro.html')

@app.route("/get")
def get_bot_response():
    return "Hola mundo"

if __name__ == '__main__':
    app.run(port=7000)
