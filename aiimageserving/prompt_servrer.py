import io
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from aiimageserving.util import set_mem_limit
from aiimageserving.generate_images import run_stable_diffusion, save_images, Precision


class Settings(BaseSettings):
    width: Optional[int] = 512
    height: Optional[int] = 512
    device: Optional[str] = "CPU:0"
    precision: Optional[Precision] = None
    batch_size: Optional[int] = 3
    mem_limit: Optional[int] = None
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
app = FastAPI()

if settings.mem_limit:
    set_mem_limit(settings.device, settings.mem_limit)

templates = Jinja2Templates("aiimageserving/static")


@app.get("/echo")
def echo(msg: str) -> str:
    return msg


@app.get("/prompt", response_class=HTMLResponse)
async def prompt_form(request: Request):
    return templates.TemplateResponse("prompt.html", {"request": request})


@app.post("/prompt",
          responses={
              200: {
                  "content": {"image/png": {}}
              }
          },
          response_class=Response)
async def prompt(req: Request) -> Response:
    """Take a prompt and use stable diffusion to generate an image.
    This is far from the ideal way of doing this as it locks up the whole web
    server while this request is being processed. To see this in action, try
    calling the `/echo` endpoint while a prompt is being processed.
    """
    if req.headers['Content-Type'] == 'application/json':
        description = (await req.json())["description"]
    else:
        description = (await req.form())["description"]

    images = run_stable_diffusion(description,
                                  settings.width,
                                  settings.height,
                                  settings.batch_size,
                                  settings.device,
                                  settings.precision)
    with io.BytesIO() as buffer:
        save_images(images, buffer)
        buffer.seek(0)
        return Response(buffer.read())
