"""
FastAPI server for image processing

Main file

@author: Stanislav Ermokhin
"""


import aiofiles
import uvicorn
import os

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from model import predict_pipeline, NEW_PICTURE_PATH
from model import __version__ as model_version


app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.resolve(strict=True) / "static"),
    name="static",
)
templates = Jinja2Templates(directory="templates")


class PredictionOutput(BaseModel):
    gender: str


@app.post("/predict")
async def upload(file: UploadFile = File(...)):
    try:
        if any(is_a_picture_extension(file, item)
               for item in
               {".png", ".jpg", ".jpeg", ".bmp", ".gif"}):

            contents = await file.read()

            index = 0
            start_filename = f"{NEW_PICTURE_PATH}/"
            end_filename = f"{file.filename}"
            filename = start_filename + f"{index}_" + end_filename

            while os.path.exists(filename):
                index += 1
                filename = start_filename + f"{index}_" + end_filename

            async with aiofiles.open(filename, "ab+") as a:
                await a.write(contents)
        else:
            raise Exception("Not a valid file format")

    except Exception:
        return {"error_message": "An error was produced",
                "model_version": model_version}

    finally:
        await file.close()

    try:
        gender = predict_pipeline(filename)
        os.remove(filename)

        return {"gender": gender,
                "model_version": model_version}

    except Exception:
        return {"error_message": "An error was produced",
                "model_version": model_version}


def is_a_picture_extension(file, extension):
    return file.filename.endswith(extension)


@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app="main:app",
                host="0.0.0.0",
                port=int(os.environ["PORT"]))
