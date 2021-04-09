import cv2
import numpy as np
import argparse

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", required=True, help="path of input (0 for webcam)")
parser.add_argument("-p", "--prototxt", required=True, help="path of Caffe 'deploy' prototxt file")
parser.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
parser.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak output")
args = vars(parser.parse_args())

# init class labels for MobileNet SSD, and then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load serialized model
print("\nLoading model...\n")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#get video for input
if(args["source"] is "0"):
    vid = cv2.VideoCapture(0) # use 0 if source should be webcam
else: vid = cv2.VideoCapture(args["source"])

# Loop through each frame
while True:

    # load the image and create blob by resizing and normalizing
    read, image = vid.read()
    if(image is None): break
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # input blob into network and return output
    print("Obtaining model output...")
    net.setInput(blob)
    output = net.forward()

    # loop over the output
    for i in np.arange(0, output.shape[2]):
        # extract the confidence of the prediction
        confidence = output[0, 0, i, 2]

        # filter out weak output by checking if the confidence is greater than the threshold
        if confidence > args["confidence"]:
            # extract the index of class label from the output, then compute the (x, y)-coord of bounding box for the object
            idx = int(output[0, 0, i, 1])
            box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[RESULT] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    #show image and wait for interrupt (if q key pressed) to complete
    cv2.imshow("Video Detector", image)

    interrupt = cv2.waitKey(1)
    if interrupt==81 or interrupt ==113:
        break

# release video feed access
vid.release()

print("\nFinished...")



# python VideoDetector.py -s "0" -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -c 0.2
# python VideoDetector.py -s "inputs/monke.mp4" -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -c 0.2



