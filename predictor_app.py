import flask
from flask import request
from predictor_api import make_prediction
from sentiment_analysis import get_paragraph_sentiment
from flask import jsonify

# Initialize the app

app = flask.Flask(__name__)

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!


@app.route("/", methods=["POST"])
def print_piped():
    if request.form['mes']:
        msg = request.form['mes']
        print(msg)
        x_input, predictions = make_prediction(str(msg))
        flask.render_template('predictor.html',
                                chat_in=x_input,
                                prediction=predictions)
    return jsonify(predictions)

@app.route("/", methods=["GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    print(request.args)
    para = request.args['chat_in'] if request.args else ''
    # para = "Has made frequent errors that are harmful to business operations. The supervisor/department head has received numerous complaints about the quality of work. The quality of work produced is unacceptable. Does not complete required paperwork. Does not require constant supervision. Error rate is acceptable, and all work is completed timely. Forms and required paperwork are completed on time with minimal errors."
    if(request.args):
        # x_input, predictions = make_prediction(request.args['chat_in'])
        x_input, predictions = get_paragraph_sentiment(para)
        print(x_input)
        no_of_sentences = len(predictions) + 1
        predictions = enumerate(predictions)
        return flask.render_template('predictor.html',
                                     chat_in=x_input,
                                     prediction=predictions,
                                     no_of_sentences=no_of_sentences)
    else: 
        #For first load, request.args will be an empty ImmutableDict type. If this is the case,
        # we need to pass an empty string into make_prediction function so no errors are thrown.
        # x_input, predictions = make_prediction('')
        x_input, predictions = get_paragraph_sentiment(para)
        no_of_sentences = len(predictions) + 1
        predictions = enumerate(predictions)
        return flask.render_template('predictor.html',
                                     chat_in=x_input,
                                     prediction=predictions,
                                     no_of_sentences=no_of_sentences)


# Start the server, continuously listen to requests.

if __name__=="__main__":
    # For local development:
    app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    app.run()
