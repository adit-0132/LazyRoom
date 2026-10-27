"""Built-in operations. Importing this package registers every operation."""
from . import adjust, effect, geometry, draw, generate  # noqa: F401
from . import itb  # noqa: F401  (ImageToolbox filter catalogue)
from . import ai_ops  # noqa: F401  (AI tools: ONNX models, downloaded on demand)
from . import shapes  # noqa: F401  (ImageToolbox crop shapes + draw line styles)
from . import texture_layer  # noqa: F401  (ImageToolbox texture generation: generate.texture)
