@app.route('/compare', methods=['POST'])
def compare():
    global reference_filename

    if reference_filename is None:
        return jsonify({"error": "Upload reference first"}), 400

    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
    img2_path = path

    result = DeepFace.verify(
        img1_path,
        img2_path,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=False,
        align=True
    )

    raw_confidence = (1 - result["distance"]) * 100
    confidence = round(raw_confidence, 2)

    print("Raw:", raw_confidence)
    print("Rounded:", confidence)

    if confidence >= 65:
        print("Branch: Strong")
        status = "Strong Match"
    elif confidence >= 45:
        print("Branch: Possible")
        status = "Possible Match"
    else:
        print("Branch: No Match")
        status = "No Match"

    return jsonify({
        "status": status,
        "confidence": round(confidence, 2)
    })