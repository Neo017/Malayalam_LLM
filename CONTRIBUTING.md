# Contributing

Please open an issue before adding a corpus, benchmark or training objective. Every contribution should document provenance, licence compatibility, Malayalam Unicode handling and whether it can contain personal or copyrighted material.

For code changes:

1. Keep dataset and model paths configurable; never hard-code credentials or private paths.
2. Add a small, deterministic test where practical.
3. Run `python -m py_compile scripts/*.py` and review the generated diff.
4. Report the exact Transformers/TRL/PEFT versions used for training changes.

Training results should include seeds, split manifests, evaluation settings and peak host/GPU memory. Do not submit raw private data, tokens, model weights, or generated content containing personal information.
