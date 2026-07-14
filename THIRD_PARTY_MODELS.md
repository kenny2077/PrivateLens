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
| [RapidOCR](https://github.com/RapidAI/RapidOCR) / `rapidocr-onnxruntime` | Local OCR | RapidOCR's project code is Apache-2.0; its upstream notice states that OCR model copyright belongs to Baidu. Verify the terms of the exact packaged model before redistribution or commercial deployment. |
| [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` | Optional face detection, embeddings, and clustering | **Important:** InsightFace code is MIT, but upstream states that its provided pretrained models, including automatically downloaded models, are for non-commercial research purposes only. The `buffalo_l` recognition weights require separate licensing for other uses; follow the upstream licensing instructions. |
| Qwen3-VL through Ollama, currently `qwen3-vl:2b-instruct-q8_0` | Optional local captions, document classification, and reranking | Review the exact Ollama artifact provenance and upstream [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). The upstream model card currently identifies Apache-2.0, but a quantized or repackaged artifact may add notices or different obligations. |

This summary was checked against upstream sources on 2026-07-13. Upstream
licenses and model artifacts can change; the source attached to the exact
download is authoritative.

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
