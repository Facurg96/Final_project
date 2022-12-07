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
import urllib.parse

app = FastAPI()

templates = Jinja2Templates(directory="./templates")

app.mount(
    "/static", StaticFiles(directory="./static"), name="static")

@app.get("/", tags=['ROOT'], response_class=HTMLResponse)
@app.get("/api/analyze/")
def index(request: Request):
    context = {"request": request}

    image_path = os.path.join("./feedback/","save.csv")
    my_products = pd.read_csv(image_path, sep=',')
    count = my_products.count()[0]

    one = my_products.iloc[count - 1]
    onee=urllib.parse.quote(one[3])
    two = my_products.iloc[count - 2]
    twoo=urllib.parse.quote(two[3])
    three = my_products.iloc[count - 3]
    threee=urllib.parse.quote(three[3])
    four = my_products.iloc[count - 4]
    fourr=urllib.parse.quote(four[3])
    return templates.TemplateResponse("index.html", {"request": request, "count": count, "one": one, "two": two, "three": three, "four": four, "onee":onee,"twoo":twoo,"threee":threee,"fourr":fourr})

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
    print(prediction)
    filename=str(prediction[0])+'.png'
    image_path = os.path.join("/static/images/",filename)
    return templates.TemplateResponse("response.html", {"request": request, "product": form_data, "prediction": prediction[0], "image_path": image_path, "score":(round(prediction[1],2)*100)})

@app.post("/api/analyze/")
async def create_item(request: Request, form_data: ProductInfo = Depends(ProductInfo.as_form)):
    data = {'name': form_data.product_name, 'description': form_data.product_description}
    prediction=model_predict(data)
    return prediction

@app.get("/")
async def root():
    return {"message": "Hello World"}

templates = Jinja2Templates(directory="templates")

@app.get("/index/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.get("/contact/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("contact.html",{"request": request})

@app.get("/sell/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("sell.html",{"request": request})

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
        new_class = str(report.get('new_class'))
        product_name = str(report.get('product_name'))
        product_description = str(report.get('product_description'))
        price = str(report.get('price'))
        prediction = str(report.get('prediction'))
        with open(file_path, 'a') as outfile:
            outfile.write(product_name)
            outfile.write(",")
            outfile.write(product_description)
            outfile.write(",")
            outfile.write(price)
            outfile.write(",")
            outfile.write(prediction)
            outfile.write(",")
            outfile.write(new_class)
            outfile.write("\n")
    return templates.TemplateResponse("thanks.html",{"request": request}) 

@app.post("/save", response_class=HTMLResponse)
async def create_item(request: Request):
    report = await request.form()
    if report != None:
        file_path=settings.SAVE_FILEPATH 
        product_name = str(report.get('product_name'))
        product_description = str(report.get('product_description'))
        price = str(report.get('price'))
        prediction = str(report.get('prediction'))
        with open(file_path, 'a') as outfile:

            outfile.write(product_name)
            outfile.write(",")
            outfile.write(product_description)
            outfile.write(",")
            outfile.write(price)
            outfile.write(",")
            outfile.write(prediction)
            outfile.write("\n")
    return templates.TemplateResponse("thanks.html",{"request": request}) 

if __name__ =="__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port=5000, reload=True)