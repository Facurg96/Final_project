"""from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dataclasses import dataclass
from fastapi import FastAPI, Form, Depends
from starlette.responses import HTMLResponse
from middleware import model_predict
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pathlib import Path

from schemas import ProductInfo

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

templates = Jinja2Templates(directory="templates")


@app.get("/index", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})


@app.get("/form", response_class=HTMLResponse)
def form_get():
    return '''<form method="post"> 
    <input type="text" name="name" value="name"/> 
    <input type="text" name="description" value="description"/> 
    <input type="submit"/> 
    </form>'''


@dataclass
class SimpleModel:
    name: str = Form(...)
    description: str = Form(...)


@app.post("/form")
def form_post(form_data: SimpleModel = Depends()):
    data = form_data
    a={'name':data.name,'description':data.description}
    prediction=model_predict(a)
    print(prediction)

    return prediction


@app.get("/", tags=['ROOT'], response_class=HTMLResponse)
def index(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("index.html", {"request": request})

def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request} )



@app.post("/index", tags=['ROOT'], response_class=HTMLResponse)    
def post_form(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    
    return templates.TemplateResponse("index.html", {"request": request}) 

@app.post("/analyze/")
async def create_item(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    print(form_data)

    return templates.TemplateResponse("response.html", {"request": request, "product": form_data}) 

@app.post("/api/analyze/")
async def create_item(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    print(form_data)
    data = {'name' :form_data.product_name, 'description': form_data.product_description}
    prediction=model_predict(data)
    print(prediction)
    
@app.get("/api/analyze/")
async def create_item(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    return templates.TemplateResponse("response.html", {"request": request, "product": form_data}) 
"""
from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from middleware import model_predict
from schemas import ProductInfo
from starlette.responses import HTMLResponse
import settings


app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount(
    "/static", StaticFiles(directory="./static"), name="static")

@app.get("/", tags=['ROOT'], response_class=HTMLResponse)
@app.get("/api/analyze/")
def index(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("index.html", {"request": request})

def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/index/", tags=['ROOT'], response_class=HTMLResponse)    
def post_form(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    
    return templates.TemplateResponse("index.html", {"request": request}) 

@app.post("/analyze/")
async def create_item(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    data = {'name': form_data.product_name, 'description': form_data.product_description}
    prediction=model_predict(data)
    # image_path = './images/' + str(prediction['prediction'][3:-3]+'.png')
    
    filename=str(prediction['prediction'][3:-3])+'.png'
    image_path = os.path.join("/app/static/images/",filename)
    return templates.TemplateResponse("response.html", {"request": request, "product": form_data, "prediction": prediction["prediction"][3:-3], "image_path": image_path})  

@app.post("/api/analyze/")
async def create_item(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    data = {'name': form_data.product_name, 'description': form_data.product_description}
    prediction=model_predict(data)
    return prediction

@app.get("/")
async def root():
    return {"message": "Hello World"}

templates = Jinja2Templates(directory="templates")

@app.get("/index", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/form", response_class=HTMLResponse)
async def create_item(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    data = {'name': form_data.product_name, 'description': form_data.product_description}
    prediction=model_predict(data)
    return templates.TemplateResponse("response.html", {"request": request, "product": form_data, "prediction": prediction}) 

@app.post("/feedback", response_class=HTMLResponse)
async def create_item(request: Request):
    report = await request.form()
    if report != None:
        file_path=settings.FEEDBACK_FILEPATH 
        
        with open(file_path, 'a') as outfile:
            outfile.write(str(report))
    return templates.TemplateResponse("index.html",{"request": request}) 

if __name__ =="__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port=5000, reload=True)