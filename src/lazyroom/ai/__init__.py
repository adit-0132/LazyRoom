"""Optional machine-learning backends (ONNX Runtime). Models are downloaded on demand, never bundled.

Modules: ``models`` (catalogue, downloads), ``runtime`` (sessions, tiling, generic
image-to-image pipeline), ``processors`` (model-specific pre/post-processing),
``ocr`` (PaddleOCR), ``segmentation`` (subject masks).  Ported from ImageToolbox
(Apache-2.0); see docs/MODELS.md and docs/dev/port_reports/ai.md.
"""
