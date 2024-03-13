build:
	docker build \
		-t transformers-rocm-docker:$(tag) \
		-f Dockerfile .

publish:
	docker image tag transformers-rocm-docker:$(tag) hardandheavy/transformers-rocm-docker:$(tag)
	docker push hardandheavy/transformers-rocm-docker:$(tag)
	docker image tag transformers-rocm-docker:$(tag) hardandheavy/transformers-rocm-docker:latest
	docker push hardandheavy/transformers-rocm-docker:latest

bash-dev:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		transformers-rocm-docker:$(tag) bash

bash:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		hardandheavy/transformers-rocm-docker:latest bash

test:
	python test/base.py
	python test/gemma-ig.py
	python test/gemma-ob.py

test-interact:
	python test/llama-interact.py

.PHONY: test
