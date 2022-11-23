from fastapi import File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel


class ProductInfo(BaseModel):
    product_name: str  
    product_description: str
    price: int
    product_image: UploadFile
    

    @classmethod 
    def as_form(
        cls,
        product_name: str = Form (...),
        product_description: str = Form (...),
        price: str = Form (...),
        product_image: UploadFile = File(...)
    ):

        if product_image.content_type != 'image/jpeg':
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Invalid format {product_image.content_type} only valid images with format jpg')
             
        return cls(
            product_name = product_name,
            product_description=product_description,
            price = price,
            product_image = product_image
    )

   
"""
class Category(BaseModel):
    category: str


lista=['2-Channel Amps',
 '2-Way Speakers',
 '2-in-1s',
 '3-Way Speakers',
 '3.5" Car Speakers',
 '360 Degree Cameras',
 '3D Blu-ray Players',
 '3D Glasses',
 '3D Printer Filament',
 '3D Printers & Filament',
 '3D Printing Accessories',
 '3D Printing Accessories & Scanners',
 '3D Scanners',
 '4" Car Speakers',
 '4" x 10" Car Speakers',
 '4" x 6" Car Speakers',
 '4G LTE Laptops',
 '4K Ultra HD Monitors']


listita= parse_obj_as(List[Category], str(lista))"""