from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # You can pass your disaggregation results to the template
    # by modifying the following line:
    results = {'appliance1': '123 kWh', 'appliance2': '456 kWh'}

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)