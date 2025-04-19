## generate pb
```bash
# use betterproto, don't use pydantic before it's compatible with v2
# move to multi stage docker build to avoid submit it to git
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=rapidocr/pb/ pb/rapidocr.proto
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=pb/ pb/rapidocr.proto
```

## local test
```bash
cd rapidocr
RAPIDOCR_GPU=false pytest -vv tests/server_test.py
```

## model download
* https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md

## configuration
| environment      | default | comment                                                              |
| ---------------- | ------- | -------------------------------------------------------------------- |
| RAPIDOCR_config  | None    | config for paddle model, default using config with package installed |
| RAPIDOCR_HOST    | 0.0.0.0 | listen host                                                          |
| RAPIDOCR_PORT    | 18910   | listen port                                                          |
| RAPIDOCR_TIMEOUT | None    | timeout int getting image content if link provided                   |
| RAPIDOCR_GPU     | true    | enable gpu or not                                                    |
