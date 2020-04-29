# HeimdallEYE USO:
# python webstreaming.py --ip 0.0.0.0 --port 8000

# Importamos paquetes o librerias necesarias
from HeimdallEYE.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# Inicializamos el marco de salida y un bloqueo utilizado para garantizar la seguridad de los hilo
# Intercambios de los marcos de salida 

outputFrame = None
lock = threading.Lock()

# Inicializamos el objeto Flask
app = Flask(__name__)

# Incializamos la transmision de video y permite que el sensor de camara sea activado
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# Retornamos el index
	return render_template("index.html")

def detect_motion(frameCount):
	# Tomamos las referencias globales a la transmision de video , el marco de salida 
	# Variables de bloqueo
	global vs, outputFrame, lock

	# Inicializamos el detector de movimiento y el numero total de fotogramas (FPS)
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# Recorremos los fps de la transmision de video
	while True:
		
		frame = vs.read()
		frame = imutils.resize(frame, width=800)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# Imprimimos en el Frame el Horario
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# Si el numero total de FPS ha alcanzado un numero suficiente
		# Para contruir un modle de fondo razonable luego se continua el procesamiento del marco
		if total > frameCount:
			# Detectamos el movimiento de la imagen
			motion = md.detect(gray)

			# Verificamos si se encontro algun movimiento en la pantalla
			if motion is not None:
				# Desempaquetamos la tupla y dibuje el cuadro que rodea el area de movimiento de la persona o objeto en la pantalla de salida
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
					(0, 0, 255), 2)
		
		# Actualizamos el modelo de fondo e incrementamos el numero total de cuadro leidos hasta aqui 
		md.update(gray)
		total += 1

		# Adquirimos el bloqueo , establecemos el marco de salida y liberamos el bloqueo
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# Tomamos referencias globables al marco de salida y bloqueamos las variables
	global outputFrame, lock

	# Bucle sobre los cuadros de la secuencia de salida
	while True:
		# Esperamos ahsta que se obtenga el bloqueo
		with lock:
			# Comprobamos si el marco de salida esta disponible , de lo contrario cancele la interacion del bucle 
			if outputFrame is None:
				continue

			# Codificamos la imagen en formato jpg
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# Nos aseguramos de que el marco se codifico correctamente
			if not flag:
				continue

		# Producimos el marco de salida en formato de bytes
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# Devolvemos la respuesta generada con los medios especificos (type (mime))
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# Comprobamos si el hilo principal esta en ejecucion
if __name__ == '__main__':
	# Contruimos el analizador de argumentos y analizamos esos argumentos de la linea de comandos ( IP , PUERTO )
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# Inicializamos un hilo que realizara la deteccion de Movimiento ( MOTION)
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	#  Arrancamos FLASK
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# Soltamos el puntero del streaming
vs.stop()
