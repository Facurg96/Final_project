from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dataclasses import dataclass
from fastapi import FastAPI, Form, Depends
from starlette.responses import HTMLResponse
from middleware import model_predict
app = FastAPI()


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
    a=dict({'name':data.name,'description':data.description})
    prediction=model_predict(a)
    print(prediction)

    return prediction