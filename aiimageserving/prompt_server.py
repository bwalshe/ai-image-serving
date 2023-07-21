import asyncio
import io
from dataclasses import dataclass
from hashlib import md5
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from aiimageserving.util import set_mem_limit
from aiimageserving.generate_images import save_images, Precision
from aiimageserving.stable_diffusion_client import StableDiffusionClient


@dataclass
class Job:
    prompt: str
    target: asyncio.Future


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

text_to_image_results = dict()
job_queue = asyncio.Queue()

if settings.mem_limit:
    set_mem_limit(settings.device, settings.mem_limit)

templates = Jinja2Templates("aiimageserving/static")

sd_runner = StableDiffusionClient()


def submit_job(prompt: str) -> str:
    job_id = md5(prompt.encode()).hexdigest()

    f = asyncio.get_event_loop().create_future()
    text_to_image_results[job_id] = f
    job_queue.put_nowait(Job(prompt, f))
    return job_id


async def worker(queue: asyncio.Queue):
    while True:
        job = await queue.get()
        job.target.set_result(await sd_runner.text_to_image(job.prompt, batch_size=settings.batch_size))
        queue.task_done()

asyncio.create_task(worker(job_queue))

@app.get("/echo", response_class=PlainTextResponse)
def echo(msg: str) -> str:
    return msg


@app.get("/text-to-image", response_class=HTMLResponse)
async def prompt_form(request: Request):
    return templates.TemplateResponse("prompt.html", {"request": request})


@app.post("/text-to-image")
async def text_to_image(req: Request) -> Response:
    """Take a prompt and use stable diffusion to generate an image.
    This is far from the ideal way of doing this as it locks up the whole web
    server while this request is being processed. To see this in action, try
    calling the `/echo` endpoint while a prompt is being processed.
    """
    if req.headers['Content-Type'] == 'application/json':
        description = (await req.json())["prompt"]
    else:
        description = (await req.form())["prompt"]

    job_id = submit_job(description)
    return RedirectResponse(f"/images/{job_id}", status_code=status.HTTP_302_FOUND)


@app.get("/images/{job_id}")
def get_images(job_id):
    if job_id not in text_to_image_results:
        raise HTTPException(status_code=404)
    try:
        images = text_to_image_results[job_id].result()
        with io.BytesIO() as buffer:
            save_images(images, buffer)
            buffer.seek(0)
            return Response(content=buffer.read(), media_type="image/png")
    except asyncio.InvalidStateError:
        return PlainTextResponse("Running")


@app.get("/queue-size")
def show_queue():
    return job_queue.qsize()
