
# render_template allows for rendering the HTML pages under /templates
# request allows for, among other things, submission of forms via POST
# redirect allows us redirect pages with code '302'
from flask import Flask, render_template, request, redirect
# Needs to be in the same directory
from tickerquery import *
from plotdata import *
import os

# Create Flask webapp object
app = Flask(__name__)

APIKEY = os.environ.get('APIKEY')

# decorator syntax - associated a URL with the python function
# @app.route('/')
# # '302' is what Flask sends back to my browser when 'redirect()' is invoked
# def hello() -> '302':
#   # return 'Hello world from Flask!'
#   # redirect to '/entry' page
#   return redirect('/entry')

# decorator syntax - associated a URL with the python function
# instead of dealing with redirects, we can associate multiple URLs to a function
@app.route('/')
@app.route('/entry')
def entry_page() -> 'html':
  # provide the name of the template, and the value to go with the_title
  return render_template('entry.html',
    the_title = 'Welcome to ticker query!')

# now this url only allows for POST method
@app.route('/display', methods = ['POST'])
def do_search() -> 'html':
  # let's use Flask's built-in object called 'request' and its dictionary attribute 'form'
  ticker = request.form['ticker']
  startdate = request.form['startdate']
  enddate = request.form['enddate']
  results = search4ticker(ticker, startdate, enddate, APIKEY)
  # results = str(search4ticker(ticker, letters))
  # invoke the log_request function:
  # log_request(request, results)
  # the display.html expects the_title and the_ticker fields - we'll provide them here
  return_comps = return_plot_html(results)
  plotjs = return_comps['script']
  plotdiv = return_comps['div']
  resources = return_comps['resources']
  return render_template('display.html',
    the_title = 'Here are your results', 
    the_ticker = ticker, the_start_date = startdate, the_end_date = enddate,
    the_resources = resources, the_data = plotjs, the_plot = plotdiv)

# Makes this app run in http://127.0.0.1:33507/
# app.run()
# If run locally, it will run in debug mode. Otherwise, the deployment will run
# its own version of app.run()
if __name__ == '__main__':
  app.run(port = 33507, debug = True)

