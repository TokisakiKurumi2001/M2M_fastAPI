from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
from starlette.background import BackgroundTask


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


@app.post("/translate/text")
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


@app.post("/translate/file")
async def translateFile(file: UploadFile = File(...), model: str = Form(...)):
    content = await file.read()
    text = content.decode('utf-8')
    _, ba_paragraphs = await pipeline(text)

    tmp_dir = 'tmp_files'
    isExist = os.path.exists(tmp_dir)
    if not isExist:
        os.makedirs(tmp_dir)
        print(f'Directory "{tmp_dir}" is created!')

    filename = str(int(time.time())) + file.filename
    with open(f'{tmp_dir}/{filename}', encoding='utf-8', mode='w') as f:
        f.write('\n'.join(ba_paragraphs))

    def cleanup(filename):
        os.remove(f'{tmp_dir}/{filename}')

    return FileResponse(
        f'{tmp_dir}/{filename}',
        background=BackgroundTask(cleanup, filename),
    )


@app.get("/models")
async def getModels():
    return {
        'models': ['M2M']
    }
