from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, Response, Request
from pydantic import BaseModel

from models import clip_model, clip_processor, translation_tokenizer, translation_model

app = FastAPI()


@app.post("/clip/image")
async def clip_image(image: UploadFile) -> list[float]:
    image_bytes = await image.read()
    image = Image.open(BytesIO(image_bytes))

    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    features = clip_model.get_image_features(**inputs)

    return features[0]


@app.get("/clip/text")
async def clip_text(text: str) -> list[float]:
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    features = clip_model.get_text_features(**inputs)

    return features[0]


@app.get("/translation")
async def translation(text: str) -> str:
    batch = translation_tokenizer(text, return_tensors="pt")
    generated_ids = translation_model.generate(batch["input_ids"])
    result = translation_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return result[0]
