from fastapi import FastAPI, File, UploadFile
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import uuid

IMAGEDIR = "images/"

model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def api_start():
    return {"Result": "Please add '/docs' after url and hit"}

@app.post("/detect-object")
async def detect_custom_object_home_result(file: bytes = File(...)):
    print(File)
    input_image = get_image_from_bytes(file)
    print("input_image =>", input_image)
    print("input_image_type =>", type(input_image))
    results = model(input_image)
    print("results =>", results)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    b = results.render()
    print("b =>", b[0])
    for img in b:
        print('img', img)
        bytes_io = io.BytesIO()
        print("bytes_io =>", bytes_io)
        img_base64 = Image.fromarray(img)
        print("img_base64 =>", img_base64)
        img_base64.save(bytes_io, format="jpeg")
        img_data = Response(content=bytes_io.getvalue(), media_type="image/jpeg")
        print("img_data",img_data)
    return img_data


@app.post("/object-to-json")
async def detect_object_return_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    print("input_image =>", input_image)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {
        "result": detect_res
    }


@app.post("/upload-object")
async def upload_image(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpeg"
    contents = await file.read()

    #save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}