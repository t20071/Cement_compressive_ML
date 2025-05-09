from wsgiref import simple_server
from flask import Flask, request, render_template, Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction

# Set environment variables for language settings
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize Flask app
app = Flask(__name__)
dashboard.bind(app)  # Bind monitoring dashboard to the app
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    # Render the home page
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    """
    Handle prediction requests.
    Accepts JSON or form data with a file path, validates input, and returns prediction results.
    """
    try:
        if request.json is not None:
            # Get file path from JSON request
            path = request.json['filepath']

            # Initialize and run prediction validation
            pred_val = pred_validation(path)
            pred_val.prediction_validation()

            # Initialize prediction object and generate predictions
            pred = prediction(path)
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)
        elif request.form is not None:
            # Get file path from form data
            path = request.form['filepath']

            # Initialize prediction object and generate predictions
            pred = prediction(path)
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    """
    Handle model training requests.
    Accepts JSON data with a folder path, validates training data, and trains the model.
    """
    try:
        if request.json['folderPath'] is not None:
            # Get folder path from JSON request
            path = request.json['folderPath']

            # Initialize and run training data validation
            train_valObj = train_validation(path)
            train_valObj.train_validation()

            # Initialize and train the model
            trainModelObj = trainModel()
            trainModelObj.trainingModel()

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

# Set server port from environment variable or default to 5001
port = int(os.getenv("PORT", 5001))

if __name__ == "__main__":
    host = '0.0.0.0'
    # Start the WSGI server
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
