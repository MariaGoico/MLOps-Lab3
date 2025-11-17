import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from pathlib import Path
import io
from PIL import Image

from logic.utilities import predict, resize

app = FastAPI(
    title="API of the Lab 1 using FastAPI",
    description="API to perform preprocessing on images",
    version="1.0.0",
)

# Serve templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the HTML homepage.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------
# Predict Endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict_class(
    class1: str = Form(..., description="First class"),
    class2: str = Form(..., description="Second class"),
    class3: str = Form(..., description="Third class"),
    class4: str = Form(..., description="Fourth class")
):
    """
    Receive exactly 4 classes for prediction.
    """
    class_list = [class1, class2, class3, class4]
    
    try:
        result = predict(class_list)
        return {"predicted_class": result}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    
# ---------------------------------------------------------
# Resize Endpoint
# ---------------------------------------------------------
@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
):
    """
    Resize an uploaded image to a random size between 28 and 225.

    The user uploads a binary image. We convert it using PIL,
    send it through the resize() logic method, and return it.
    """

    try:
        # Read file bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save the uploaded image temporarily
        input_path = Path("temp_input.jpg")
        input_path.write_bytes(contents)

        # Apply random resize using your utilities
        resized_img = resize(str(input_path))

        # Save to a bytes buffer to return
        buf = io.BytesIO()
        resized_img.save(buf, format="JPEG")
        buf.seek(0)

        return FileResponse(
            buf,
            media_type="image/jpeg",
            filename="resized.jpg",
        )

    except Exception as e:
        return {"error": f"Could not process image: {str(e)}"}
    

# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)