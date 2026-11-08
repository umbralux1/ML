.PHONY: install-model delete-model

install-model:
	./venv/bin/hf download Qwen/Qwen2.5-7B-Instruct

delete-model:
	./venv/bin/hf cache rm model/Qwen/Qwen2.5-7B-Instruct
