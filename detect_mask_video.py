
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from pygame import mixer

def detect_and_predict_mask(frame, faceNet, maskNet):
	# construct a frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# obtaining the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# list of faces, their corresponding locations, and the list of predictions
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# preprocessing the image and extract the face ROI
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# appending the face and bounding boxes
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# loading serialized face detector
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[STAGE] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect wearing face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		if(label=="No Mask"):
			mixer.init()
			sound = mixer.Sound(r"audio_sample\buzzer-2.wav")
			sound.play()
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# break from the loop
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()