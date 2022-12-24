from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware


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
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    b = results.render()
    print("b =>", b)
    for img in b:
        print('img', img)
        bytes_io = io.BytesIO()
        print("bytes_io =>", bytes_io)
        img_base64 = Image.fromarray(img)
        print("img_base64 =>", img_base64)
        img_base64.save(bytes_io, format="jpeg")
        img_data = Response(content=bytes_io.getvalue(), media_type="image/jpeg")
        print("img_data",img_data)
    # return {
    #     "result": detect_res
    # }

    return img_data


@app.post("/object-to-json")
async def detect_food_return_base64_img():
    # input_image = get_image_from_bytes(file)
    # print("input_image =>", input_image)
    # results = model(input_image)
    # print("results =>", results)
    # a = results.render()  # updates results.imgs with boxes and labels
    # print("a =>", a)
    # for img in a:
    #     print('img', img)
    #     bytes_io = io.BytesIO()
    #     img_base64 = Image.fromarray(img)
    #     img_base64.save(bytes_io, format="jpeg")
    image_value = detect_res
    return {
        "result": detect_res
    }
