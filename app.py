import json
from datetime import datetime
from flask import Flask, render_template, request, abort
from ultralytics import YOLO
import os
import csv
import cv2

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load YOLO Model
model = YOLO("model/best.pt")


def save_report_data(
        filename,
        pothole_count,
        severe_count,
        medium_count,
        minor_count,
        confidence,
        road_health_score):
    """Save report data as JSON so the /report route can render it as HTML."""

    report_id = f"RDS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    date_str  = datetime.now().strftime("%d %B %Y, %H:%M")

    data = {
        "report_id":         report_id,
        "filename":          filename,
        "date":              date_str,
        "pothole_count":     pothole_count,
        "severe_count":      severe_count,
        "medium_count":      medium_count,
        "minor_count":       minor_count,
        "confidence":        confidence,
        "road_health_score": road_health_score,
    }

    json_path = os.path.join(
        REPORT_FOLDER,
        f"{os.path.splitext(filename)[0]}.json"
    )

    with open(json_path, "w") as f:
        json.dump(data, f)

    # Return the report URL (used in the template button)
    return f"report/{os.path.splitext(filename)[0]}"


@app.route("/report/<name>")
def view_report(name):
    """Render the beautiful HTML report page."""
    json_path = os.path.join(REPORT_FOLDER, f"{name}.json")

    if not os.path.exists(json_path):
        abort(404)

    with open(json_path, "r") as f:
        data = json.load(f)

    return render_template("report.html", **data)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_path = None
    filename = None

    pothole_count = 0
    severe_count = 0
    medium_count = 0
    minor_count = 0

    pdf_report = None
    road_health_score = 100

    total_images_tested = len(os.listdir(UPLOAD_FOLDER))

    try:
        with open("model/accuracy.txt", "r") as f:
            model_accuracy = f.read().strip() + "%"
    except:
        model_accuracy = "Not Available"

    if request.method == "POST":

        print("POST REQUEST RECEIVED")
        print(request.files)

        file = request.files.get("image")

        if file and file.filename != "":

            filename = file.filename

            filepath = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(filepath)

            total_images_tested = len(
                os.listdir(UPLOAD_FOLDER)
            )

            results = model(
                filepath,
                conf=0.70,
                iou=0.35
            )

            img_cv = cv2.imread(filepath)

            total_confidence = 0

            for result in results:

                for box in result.boxes:

                    pothole_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    conf = float(box.conf[0])

                    total_confidence += conf

                    if conf >= 0.85:

                        severity = "Severe"
                        color = (0, 0, 255)
                        severe_count += 1

                    elif conf >= 0.75:

                        severity = "Medium"
                        color = (0, 165, 255)
                        medium_count += 1

                    else:

                        severity = "Minor"
                        color = (0, 255, 255)
                        minor_count += 1

                    cv2.rectangle(
                        img_cv,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3
                    )

                    cv2.putText(
                        img_cv,
                        f"{severity} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

            if pothole_count > 0:

                prediction = "Pothole Detected"

                confidence = round(
                    (total_confidence / pothole_count) * 100,
                    2
                )

            else:

                prediction = "Normal Road"
                confidence = 100.0

            road_health_score = 100

            road_health_score -= (severe_count * 20)
            road_health_score -= (medium_count * 10)
            road_health_score -= (minor_count * 5)

            if road_health_score < 0:
                road_health_score = 0

            result_path = os.path.join(
                RESULT_FOLDER,
                filename
            )

            cv2.imwrite(
                result_path,
                img_cv
            )

            image_path = result_path

            # Save report data and get report URL
            pdf_report = save_report_data(
                filename,
                pothole_count,
                severe_count,
                medium_count,
                minor_count,
                confidence,
                road_health_score
            )

            file_exists = os.path.exists(
                "prediction_history.csv"
            )

            with open(
                "prediction_history.csv",
                "a",
                newline=""
            ) as file_csv:

                writer = csv.writer(file_csv)

                if not file_exists:

                    writer.writerow([
                        "Image Name",
                        "Result",
                        "Confidence"
                    ])

                writer.writerow([
                    filename,
                    prediction,
                    f"{confidence}%"
                ])

        history = []

        try:

            with open(
                "prediction_history.csv",
                "r"
            ) as file_csv:

                reader = csv.reader(file_csv)

                next(reader)

                for row in reader:
                    history.append(row)

        except:
            pass

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            pothole_count=pothole_count,
            severe_count=severe_count,
            medium_count=medium_count,
            minor_count=minor_count,
            image_path=image_path,
            filename=filename,
            total_images_tested=total_images_tested,
            model_accuracy=model_accuracy,
            history=history,
            pdf_report=pdf_report,
            road_health_score=road_health_score
        )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        pothole_count=pothole_count,
        severe_count=severe_count,
        medium_count=medium_count,
        minor_count=minor_count,
        image_path=image_path,
        filename=filename,
        total_images_tested=total_images_tested,
        model_accuracy=model_accuracy,
        history=[],
        pdf_report=pdf_report,
        road_health_score=road_health_score
    )


# ==========================
# LIVE CAMERA DETECTION
# ==========================

@app.route("/live_detection")
def live_detection():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=0.70)

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                if conf >= 0.85:
                    color = (0, 0, 255)
                    severity = "Severe"

                elif conf >= 0.75:
                    color = (0, 165, 255)
                    severity = "Medium"

                else:
                    color = (0, 255, 255)
                    severity = "Minor"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                cv2.putText(
                    frame,
                    f"{severity} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        cv2.imshow(
            "Live Pothole Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return """
    <h2>Live Detection Closed</h2>
    <a href="/">Back to Home</a>
    """

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
