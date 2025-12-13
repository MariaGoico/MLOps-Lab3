import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi import Request
from pathlib import Path
import io
from PIL import Image
from PIL import UnidentifiedImageError

from logic.utilities import predict, resize


app = FastAPI(
    title="API of the Lab 1 using FastAPI",
    description="API to perform preprocessing on images",
    version="1.0.0",
)

# Serve templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")



# Home Page

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the HTML homepage.
    """
    return templates.TemplateResponse(request, "index.html", {"request": request})



# Predict Endpoint

@app.post("/predict")
async def predict_class(
    file: UploadFile = File(...),
):
    """
    Predict the class for an uploaded pet image using the trained ONNX model.

    Args:
        file: The image file to classify

    Returns:
        dict: The predicted class and confidence (if available)
    """
    try:
        # Read the uploaded file to validate it's an image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Predict using the ONNX model
        result, confidence = predict(image)

        response = {"predicted_class": result, "filename": file.filename}

        # Add confidence if available
        if confidence is not None:
            response["confidence"] = round(confidence, 4)

        return response

    except UnidentifiedImageError:
        return {"error": "Uploaded file is not a valid image."}

    except OSError as e:# pragma: no cover
        return {"error": f"Failed to read the image: {str(e)}"}



# Resize Endpoint

@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Form(None),
    height: int = Form(None),
):
    """
    Resize an uploaded image to specified dimensions.
    If width or height are not provided, random size between 28-225 is used.

    Args:
        file: The image file to resize
        width: Target width (optional, random if not provided)
        height: Target height (optional, random if not provided)

    Returns:
        StreamingResponse: The resized image as JPEG
    
    Rules from tests:
    - Negative dimensions -> return JSON describing the error (status 200!)
    - Only valid (positive) dimensions -> return resized image
    - Missing dimension -> random value allowed
    """

    # Case 1: Invalid width/height => must return JSON (NOT an image)
    invalid_info = {}

    if width is not None and width <= 0:
        invalid_info["width"] = "Width must be greater than 0"
    if height is not None and height <= 0:# pragma: no cover
        invalid_info["height"] = "Height must be greater than 0"

    if invalid_info:
        # The test expects status 200 and JSON
        return {"error": invalid_info}

    # Read image directly
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # If your resize function needs a path, create temp file properly
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            image.save(tmp.name)
            img = resize(tmp.name, width=width, height=height)
        
        # Or if resize can take a PIL Image directly:
        # img = resize(image, width=width, height=height)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")
        
    except UnidentifiedImageError:# pragma: no cover
        return {"error": "Uploaded file is not a valid image."}
    except Exception as e: # pylint: disable=broad-exception-caught # pragma: no cover
        return {"error": f"Failed to process image: {str(e)}"}



# Get Output File Endpoint (Optional - to retrieve saved files)

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """
    Retrieve a file from the outputs directory.

    Args:
        filename: Name of the file to retrieve

    Returns:
        FileResponse: The requested file
    """
    file_path = Path("outputs") / filename

    if not file_path.exists():
        return {"error": "File not found"}

    return FileResponse(file_path)


# Entry point (for direct execution only)
if __name__ == "__main__":  # pragma: no cover
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
