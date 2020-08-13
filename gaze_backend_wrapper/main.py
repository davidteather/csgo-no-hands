from flask import Flask, request, jsonify, send_file
from selenium import webdriver
import io
import base64
#import logging
from datauri import DataURI
application = Flask(__name__)

driver = webdriver.Chrome()
driver.get("https://webgazer.cs.brown.edu/calibration.html?")
print("Calibrate")

#
# User Likes By Username
#
@application.route('/eye', methods=["GET"])
def eyePosition():
    #j = driver.execute_script("webgazer.getCurrentPrediction().then((data) => { return data } )")
    j = driver.execute_script("return webgazer.getCurrentPrediction()")
    print(j)
    return jsonify(j)

@application.route('/frame', methods=["GET"])
def frame():
    j = driver.execute_script("return document.getElementById('webgazerVideoCanvas').toDataURL()")
    uri = DataURI(j)
    return send_file(io.BytesIO(uri.data), mimetype=uri.mimetype)

@application.route("/test", methods=["GET"])
def y():
    return "k"
if __name__ == '__main__':
    application.run(debug=False, port=5005)