from typing import Optional

from fastapi import File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Required


class ProductInfo(BaseModel):
    product_name: str  
    product_description: str
    price: str
    product_image: UploadFile
    

    @classmethod 
    def as_form(
        cls,
        product_name: str = Form (...),
        product_description: str = Form (...),
        price: str = Form (...),
        product_image: UploadFile = File (...)
    ):
        if product_image:
            if product_image.content_type != 'image/jpeg':
                raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid format {product_image.content_type} only valid images with format jpg')
        else:
            pass
             
        return cls(
            product_name = product_name,
            product_description = product_description,
            price = price,
            product_image = product_image
    )

   