import asyncio
import logging
from time import time
from typing import List
from typing import Optional

from grpclib.const import Status
from grpclib.exceptions import GRPCError
from grpclib.health.service import Health
from grpclib.reflection.service import ServerReflection
from grpclib.server import Server
from grpclib.utils import graceful_exit
import httpx
from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import format_html
import PIL
from pydantic import Field
from pydantic import PositiveInt
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

from pb.rapidocr import Image
from pb.rapidocr import Part
from pb.rapidocr import Point
from pb.rapidocr import RapidOcrServiceBase
from pb.rapidocr import Response
from pb.rapidocr import TableResponse

# TODO(Deo): paddle mess up root logging, use this to reset
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for handler in logger.handlers:
  handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  )


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_prefix='rapidocr_')
  config: Optional[str] = Field(
    None,
    description='config file path, missing use default small models',
    examples=['./config.yaml'],
  )
  host: str = Field('0.0.0.0', description='listen host')
  port: int = Field(18910, description='listen port')
  timeout: Optional[PositiveInt] = Field(
    None, description='timeout in getting image from link'
  )
  gpu: bool = Field(True, description='use gpu or not')


class RapidOCRService(RapidOcrServiceBase):
  _wired_engine: WiredTableRecognition
  _lineless_engine: LinelessTableRecognition
  _table_cls: TableCls

  @property
  def wired_engine(self) -> WiredTableRecognition:
    if not hasattr(self, '_wired_engine'):
      self._wired_engine = WiredTableRecognition()
    return self._wired_engine

  @property
  def lineless_engine(self) -> LinelessTableRecognition:
    if not hasattr(self, '_lineless_engine'):
      self._lineless_engine = LinelessTableRecognition()
    return self._lineless_engine

  @property
  def table_cls(self) -> TableCls:
    if not hasattr(self, '_table_cls'):
      self._table_cls = TableCls('cpu' if not self.gpu else 'cuda')
    return self._table_cls

  def __init__(self, config: str, timeout: Optional[PositiveInt], gpu: bool):
    if gpu:
      from rapidocr_paddle import RapidOCR
    else:
      from rapidocr_onnxruntime import RapidOCR
    self.engine = RapidOCR(
      config_path=config, det_use_cuda=gpu, cls_use_cuda=gpu, rec_use_cuda=gpu
    )
    self.timeout = timeout
    self.gpu = gpu

  async def _recognize(self, request: Image):
    request.box_thresh = (
      request.box_thresh if request.box_thresh is not None else 0.5
    )
    request.text_score = (
      request.text_score if request.text_score is not None else 0.5
    )
    request.unclip_ratio = (
      request.unclip_ratio if request.unclip_ratio is not None else 1.6
    )
    if request.data:
      # image in raw bytes
      content = request.data
    elif request.link:
      # image url address
      res = httpx.get(request.link, timeout=self.timeout)
      if not res.is_success:
        raise GRPCError(Status.INTERNAL, 'failed to get image from link')
      content = res.content
      request.data = content
    else:
      raise GRPCError(Status.INVALID_ARGUMENT, 'no image data or link')
    try:
      return self.engine(
        content,
        use_det=request.use_det,
        use_cls=request.use_cls,
        use_rec=request.use_rec,
        box_thresh=request.box_thresh,
        text_score=request.text_score,
        unclip_ratio=request.unclip_ratio,
      )
    except MemoryError:
      logger.exception('failed to recognize image')
      raise GRPCError(Status.RESOURCE_EXHAUSTED, 'out of memory')
    except PIL.UnidentifiedImageError:
      logger.warning('failed to recognize image')
      raise GRPCError(Status.INVALID_ARGUMENT, 'invalid image format')

  async def recognize(self, request: Image) -> Response:
    start = time()
    info = request.info or request.link or 'data'
    logger.info('recieved %s', info)
    result, _ = await self._recognize(request)
    res = Response()
    if not result:
      logger.warning(
        'failed to recognize %s in %.3f ms', info, (time() - start) * 1000
      )
      return res
    for i in result:
      points, text, confidence = i
      part = Part(text=text, confidence=confidence)
      res.parts.append(part)
      for x, y in points:
        part.points.append(Point(x=x, y=y))
    logger.info(
      'finish recognizing %s in %.3f ms', info, (time() - start) * 1000
    )
    return res

  async def recognize_table(self, request: Image) -> TableResponse:
    start = time()
    info = request.info or request.link or 'data'
    logger.info('recieved %s', info)
    result, _ = await self._recognize(request)

    cls, elasp = self.table_cls(request.data)
    if cls == 'wired':
      table_engine = self.wired_engine
    else:
      table_engine = self.lineless_engine

    html, elasp2, _, _, _ = table_engine(request.data, ocr_result=result)
    res = TableResponse(html=format_html(html))
    logger.info(
      'finish recognizing table %s in %.3f/%.3f/%.3f ms',
      info,
      elasp,
      elasp2,
      (time() - start) * 1000,
    )
    return res


def get_services(*args, **kwargs) -> List:
  return ServerReflection.extend([RapidOCRService(*args, **kwargs), Health()])


async def serve(settings: Settings):
  server = Server(
    get_services(settings.config, timeout=settings.timeout, gpu=settings.gpu)
  )
  with graceful_exit([server]):
    await server.start(settings.host, settings.port)
    logger.info(
      'listen on %s:%d, using %s',
      settings.host,
      settings.port,
      settings.config or 'default small models',
    )
    await server.wait_closed()
    logging.info('Goodbye!')


if __name__ == '__main__':
  _settings = Settings()
  asyncio.run(serve(_settings))
