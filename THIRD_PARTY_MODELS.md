# Third-Party Models and Licenses

The [MIT License](LICENSE) applies to PrivateLens source code. It does not
relicense Python dependencies, model runtimes, training data, pretrained
weights, quantizations, or services.

The PrivateLens wheel does not bundle the pretrained weights listed below.
Enabled extractors may download weights into a local cache or call a separately
installed local runtime such as Ollama. You are responsible for confirming the
terms of the exact artifact you download and for complying with them.

## Current model/runtime inventory

| Component | PrivateLens use | License action required |
|-----------|-----------------|-------------------------|
| [OpenCLIP](https://github.com/mlfoundations/open_clip) with `ViT-B-32-quickgelu` / `openai` weights | Semantic image and text embeddings | Review the OpenCLIP and [OpenAI CLIP](https://github.com/openai/CLIP) repositories and the exact downloaded model card. PrivateLens grants no additional rights to the implementation, weights, or training data. |
| [RapidOCR](https://github.com/RapidAI/RapidOCR/tree/v1.4.4) / `rapidocr-onnxruntime==1.4.4` | Local OCR | The locked wheel and v1.4.4 project declare Apache-2.0, while upstream identifies Baidu as the OCR-model copyright owner. The model archive has no model-specific license, so the published GHCR image excludes this dependency. |
| [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` | Optional face detection, embeddings, and clustering | **Important:** InsightFace code is MIT, but upstream states that its provided pretrained models, including automatically downloaded models, are for non-commercial research purposes only. The `buffalo_l` recognition weights require separate licensing for other uses; follow the upstream licensing instructions. |
| Qwen3-VL through Ollama, currently `qwen3-vl:2b-instruct-q8_0` | Optional local captions, document classification, and reranking | Review the exact Ollama artifact provenance and upstream [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). The upstream model card currently identifies Apache-2.0, but a quantized or repackaged artifact may add notices or different obligations. |

This summary was checked against upstream sources on 2026-07-14. Upstream
licenses and model artifacts can change; the source attached to the exact
download is authoritative.

## RapidOCR redistribution record

The optional `full`/`ml` install resolves `rapidocr-onnxruntime==1.4.4`. Its
published wheel declares `License: Apache-2.0`, and the corresponding upstream
[v1.4.4 license](https://github.com/RapidAI/RapidOCR/blob/v1.4.4/LICENSE)
applies Apache-2.0 to the project. The upstream README separately preserves
Baidu's copyright in the OCR models. The model source, PaddleOCR, is also
[released under Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE).

The locked wheel is
`rapidocr_onnxruntime-1.4.4-py3-none-any.whl` with SHA-256
`971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf`.
It embeds these unmodified ONNX files:

| File | SHA-256 |
|------|---------|
| `ch_PP-OCRv4_det_infer.onnx` | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` |
| `ch_PP-OCRv4_rec_infer.onnx` | `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b` |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

The PrivateLens wheel does not contain these files. The v1.0 GHCR artifact is
an explicitly suffixed `core` image and does not install the ML dependencies or
these ONNX models. A locally built `full` image installs the unmodified wheel
and includes this record, PrivateLens's MIT license, and the RapidOCR project's
exact v1.4.4 license under `/licenses/RapidOCR-LICENSE.txt`; that attribution does not establish a
separate license grant for the Baidu-copyrighted model files. Do not
redistribute a locally built full image until the exact model rights have been
confirmed for the intended use.

This provenance review was completed against primary upstream sources and the
locked wheel on 2026-07-14. It records the artifacts and known notices but is
not legal advice.

RapidOCR 1.4.4 declares Python `<3.13`. The PrivateLens core supports Python
3.11–3.13, while the complete `full` and `ml` stacks are release-gated only on
Python 3.11 in v1.0. Their locked dependencies resolve on Python 3.12, but that
full runtime is not verified there. The model-free core container uses 3.11.

## InsightFace commercial-use warning

Do not assume that installing `privatelens[full]` makes the default face model
commercially usable. Face extraction is opt-in, reads `settings.face_model`,
and currently defaults to `buffalo_l`; PrivateLens's MIT license covers the
integration code, not those weights. For commercial, organizational, or
public-service deployment, either obtain the required upstream license or
configure a face model whose terms permit the intended use. Keep face
extraction disabled until that review is complete.

## Privacy and model downloads

- Model downloads reveal normal network metadata to the model host even though
  user photos are not intentionally uploaded.
- PrivateLens defaults its application model cache to `~/.privatelens/models/`;
  individual runtimes may also use their own documented cache locations.
- Review model caches before copying or redistributing a PrivateLens data
  directory or container volume.
- Keep Ollama on loopback or a trusted private network. A remote Ollama endpoint
  can receive image content when VLM features are used.
- Face embeddings and derived captions can remain sensitive even when the
  original photo is absent.

## Requirements for changing a model

A contribution that adds or changes a model must document:

1. exact model and revision or tag;
2. download source and integrity/provenance information;
3. implementation and weight licenses;
4. commercial-use, attribution, redistribution, and acceptable-use limits;
5. training-data or biometric risks known from the upstream documentation;
6. cache location and removal procedure;
7. CPU/GPU, memory, and platform expectations;
8. a generated-data verification path.

If the terms are unclear, do not make the model a default.
