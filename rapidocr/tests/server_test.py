from functools import partial

from grpclib.const import Status
from grpclib.exceptions import GRPCError
from grpclib.health.v1.health_grpc import HealthStub
from grpclib.health.v1.health_pb2 import HealthCheckRequest
from grpclib.health.v1.health_pb2 import HealthCheckResponse
from grpclib.reflection.v1.reflection_grpc import ServerReflectionStub
from grpclib.reflection.v1.reflection_pb2 import ServerReflectionRequest
from grpclib.testing import ChannelFor
import pytest
from server import get_services
from server import Settings

from pb.rapidocr import Image
from pb.rapidocr import RapidOcrServiceStub

approx = partial(pytest.approx, abs=1e-5)


@pytest.fixture(scope='module', autouse=True)
def anyio_backend():
  return 'asyncio'


async def test_rapidocr_service():
  settings = Settings()
  services = get_services(settings.config, settings.timeout, settings.gpu)
  async with ChannelFor(services) as channel:
    stub = RapidOcrServiceStub(channel)
    health = HealthStub(channel)
    reflection = ServerReflectionStub(channel)

    with pytest.raises(GRPCError, match='no image data or link') as e:
      await stub.recognize(Image())
    assert e.value.status == Status.INVALID_ARGUMENT

    with pytest.raises(GRPCError, match='invalid image format') as e:
      await stub.recognize(Image(data=b'abc'))
    assert e.value.status == Status.INVALID_ARGUMENT

    # recognize table
    with open('tests/moonshot_price.jpg', 'rb') as f:
      data = f.read()

    response = await stub.recognize_table(Image(data=data))
    with open('tests/moonshot_price.html') as f:
      assert response.html == f.read()

    # recognize
    with open('tests/id_card.jpg', 'rb') as f:
      data = f.read()
    response = await stub.recognize(Image(data=data))
    assert response.to_dict() == {
      'parts': [
        {
          'points': [
            {'x': 46.0, 'y': 50.0},
            {'x': 199.0, 'y': 50.0},
            {'x': 199.0, 'y': 79.0},
            {'x': 46.0, 'y': 79.0},
          ],
          'text': '姓名张大民',
          'confidence': approx(0.9745097160339355),
        },
        {
          'points': [
            {'x': 48.0, 'y': 98.0},
            {'x': 256.0, 'y': 98.0},
            {'x': 256.0, 'y': 122.0},
            {'x': 48.0, 'y': 122.0},
          ],
          'text': '性别男名旅汉',
          'confidence': approx(0.9919305443763733),
        },
        {
          'points': [
            {'x': 47.0, 'y': 143.0},
            {'x': 304.0, 'y': 145.0},
            {'x': 304.0, 'y': 167.0},
            {'x': 47.0, 'y': 165.0},
          ],
          'text': '出生1986年10月20日',
          'confidence': approx(0.9658491611480713),
        },
        {
          'points': [
            {'x': 48.0, 'y': 192.0},
            {'x': 121.0, 'y': 190.0},
            {'x': 121.0, 'y': 211.0},
            {'x': 48.0, 'y': 213.0},
          ],
          'text': '住址',
          'confidence': approx(0.7932530641555786),
        },
        {
          'points': [
            {'x': 115.0, 'y': 193.0},
            {'x': 310.0, 'y': 193.0},
            {'x': 310.0, 'y': 213.0},
            {'x': 115.0, 'y': 213.0},
          ],
          'text': '上海市黄浦区广东路',
          'confidence': approx(0.9945125579833984),
        },
        {
          'points': [
            {'x': 114.0, 'y': 220.0},
            {'x': 171.0, 'y': 220.0},
            {'x': 171.0, 'y': 245.0},
            {'x': 114.0, 'y': 245.0},
          ],
          'text': '689号',
          'confidence': approx(0.9895161390304565),
        },
        {
          'points': [
            {'x': 46.0, 'y': 304.0},
            {'x': 515.0, 'y': 304.0},
            {'x': 515.0, 'y': 328.0},
            {'x': 46.0, 'y': 328.0},
          ],
          'text': '公民身份号码310101198610203222',
          'confidence': approx(0.9965057969093323),
        },
      ]
    }

    # health
    response = await health.Check(HealthCheckRequest())
    assert response.status == HealthCheckResponse.SERVING

    # reflection
    response = await reflection.ServerReflectionInfo(
      [ServerReflectionRequest(file_containing_symbol='RAPIDOCR')]
    )
    assert len(response) == 1
    # TODO(Deo): it's not found at the moment
    #   https://github.com/danielgtaylor/python-betterproto/issues/443
    # assert response[0].name == ''
    # assert response[0].package == ''
