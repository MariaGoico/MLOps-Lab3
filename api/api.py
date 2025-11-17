from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import tempfile
from logic.utilities import predict, resize

app = FastAPI(title="Image Processing API", version="1.0")


# ─────────────────────────────
# PREDICT ENDPOINT
# ─────────────────────────────
@app.post("/predict")
async def predict_endpoint(classes: list[str]):
    """
    Predict a random class from a provided list.
    """
    result = predict(classes)
    return {"predicted_class": result}


# ─────────────────────────────
# RESIZE ENDPOINT
# ─────────────────────────────
@app.post("/resize")
async def resize_endpoint(file: UploadFile = File(...)):
    """
    Upload an image → resize → return resized image.
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input:
        temp_input.write(await file.read())
        temp_input_path = temp_input.name

    # Resize the image
    resized_image = resize(temp_input_path)

    # Save resized image to temporary output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_output:
        resized_image.save(temp_output.name)
        temp_output_path = temp_output.name

    return FileResponse(temp_output_path, media_type="image/png")
