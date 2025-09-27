from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os
import gdown
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# =================== Model Setup ===================
MODEL_PATH = "./models/jossa-dummy.joblib"
FILE_ID = "1rQsmDvPoqFjAGNd0qYpj8CCoyL1wP2QQ"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# Lazy-load model
model = None
def get_model():
    global model
    if model is None:
        print("Loading ML model...")
        model = joblib.load(MODEL_PATH)
    return model

# =================== Dataset & Constants ===================
df = pd.read_csv("2021.csv")  # optional

colleges = [
    "National Institute of Technology, Uttarakhand",
    "National Institute of Technology, Warangal",
    "Sardar Vallabhbhai National Institute of Technology, Surat",
    "Visvesvaraya National Institute of Technology, Nagpur",
    "National Institute of Technology, Andhra Pradesh",
    "Indian Institute of Engineering Science and Technology, Shibpur",
    "National Institute of Technology, Arunachal Pradesh",
    "National Institute of Technology, Jamshedpur",
    "National Institute of Technology, Kurukshetra",
    "National Institute of Technology, Manipur",
    "National Institute of Technology, Mizoram",
    "National Institute of Technology, Rourkela",
    "National Institute of Technology, Silchar",
    "National Institute of Technology, Srinagar",
    "National Institute of Technology, Tiruchirappalli",
    "Dr. B R Ambedkar National Institute of Technology, Jalandhar",
    "Malaviya National Institute of Technology, Jaipur",
    "Maulana Azad National Institute of Technology, Bhopal",
    "Motilal Nehru National Institute of Technology, Allahabad",
    "National Institute of Technology, Agartala",
    "National Institute of Technology, Calicut",
    "National Institute of Technology, Delhi",
    "National Institute of Technology, Durgapur",
    "National Institute of Technology, Goa",
    "National Institute of Technology, Hamirpur",
    "National Institute of Technology Karnataka, Surathkal",
    "National Institute of Technology, Meghalaya",
    "National Institute of Technology, Nagaland",
    "National Institute of Technology, Patna",
    "National Institute of Technology, Puducherry",
    "National Institute of Technology, Raipur",
    "National Institute of Technology, Sikkim"
]

programs = [
    "Computer Science and Engineering (4 Years, B.Tech)",
    "Electronics and Communication Engineering (4 Years, B.Tech)",
    "Mechanical Engineering (4 Years, B.Tech)",
    "Civil Engineering (4 Years, B.Tech)",
    "Electrical Engineering (4 Years, B.Tech)",
    "Chemical Engineering (4 Years, B.Tech)",
    "Electrical and Electronics Engineering (4 Years, B.Tech)",
    "Metallurgical and Materials Engineering (4 Years, B.Tech)",
    "Information Technology (4 Years, B.Tech)",
    "Computer Engineering (4 Years, B.Tech)",
]

# =================== Routes ===================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    user_rank = int(request.form["Rank"])
    seat_type = request.form["Seat_Type"]
    quota = request.form["Quota"]
    gender = request.form["Gender"]
    pwd = request.form["PWD"]
    year = int(request.form["Year"])

    results = []

    for institute in colleges:
        for program in programs:
            input_data = {
                "Institute": institute,
                "Academic Program Name": program,
                "Quota": quota,
                "Seat Type": seat_type,
                "Gender": gender,
                "Year": year,
                "PWD": pwd
            }
            features_df = pd.DataFrame([input_data])
            predicted_rank = get_model().predict(features_df)[0]

            if user_rank - 2000 <= predicted_rank:
                results.append({
                    "Institute": institute,
                    "Program": program,
                    "Predicted Closing Rank": int(predicted_rank)
                })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return render_template("home.html", tables=[], user_rank=user_rank,
                               message="No eligible options found.")

    # Deduplicate and sort
    results_df = results_df.sort_values("Predicted Closing Rank").drop_duplicates(
        subset=["Institute", "Program"], keep="first"
    ).reset_index(drop=True)

    results_df.to_csv("eligible_results.csv", index=False)

    return render_template(
        "home.html",
        tables=[results_df.to_html(classes='table table-striped', index=False)],
        user_rank=user_rank,
        message=f"Found {len(results_df)} eligible options."
    )

@app.route("/download")
def download_pdf():
    df = pd.read_csv("eligible_results.csv")
    pdf_file = "eligible_colleges.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)

    styles = getSampleStyleSheet()
    elements = [Paragraph("Eligible Colleges & Branches", styles['Heading1'])]
    data = [df.columns.tolist()] + df.values.tolist()

    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 8)
    ]))

    elements.append(table)
    doc.build(elements)

    return send_file(pdf_file, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
