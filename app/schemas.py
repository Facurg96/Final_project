from typing import Optional

from fastapi import File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Required


class ProductInfo(BaseModel):
    product_name: str  
    product_description: str
    price: str
    

    @classmethod 
    def as_form(
        cls,
        product_name: str = Form (...),
        product_description: str = Form (...),
        price: str = Form (...),

    ):

             
        return cls(
            product_name = product_name,
            product_description = product_description,
            price = price,
    
    )

   