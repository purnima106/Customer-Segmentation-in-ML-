from flask import *
import pickle
import os

model_path = os.path.join(os.getcwd(), "pc.model")

f = None
try:
    f = open(model_path, "rb")
    model = pickle.load(f)
except Exception as e:
    print("Issue:", e)
finally:
    if f is not None:
        f.close()

app = Flask(__name__)

@app.route("/")
def home():
    age = request.args.get("age")
    income = request.args.get("income")
    spending = request.args.get("spending")
    
    if age and income and spending:
        try:
            age = float(age)
            income = float(income)
            spending = float(spending)
            data = [[age, spending, income]]
            satisfaction = model.predict(data)
            if satisfaction == 0:
                msg = "Customer has SILVER MEMBERSHIP."
            elif satisfaction == 1:
                msg = "Customer has BRONZE MEMBERSHIP."
            else:
                msg = "Customer has GOLD MEMBERSHIP."
        except ValueError:
            msg = "Invalid input. Please enter numeric values for age, income, and spending."
    else:
        msg = ""

    return render_template("home.html", msg=msg)
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
