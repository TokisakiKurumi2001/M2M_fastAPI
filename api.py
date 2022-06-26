from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import TranslationPipeline


class TranslationItem(BaseModel):
    text: str
    model: str


pipeline = TranslationPipeline()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/translate")
async def translate(translation: TranslationItem):
    print(f"Khoi_M2M, {translation.text[0:50]}")
    text = translation.text
    vi_paragraphs, ba_paragraphs = await pipeline(text)
    return {
        'IsSuccessed': True,
        'Message': 'Success',
        'ResultObj': {
            'src': vi_paragraphs,
            'tgt': ba_paragraphs
        }
    }
