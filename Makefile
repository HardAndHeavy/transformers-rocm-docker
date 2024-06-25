build:
	docker build \
		-t transformers-rocm:$(tag) \
		-f Dockerfile .

publish:
	docker image tag transformers-rocm:$(tag) hardandheavy/transformers-rocm:$(tag)
	docker push hardandheavy/transformers-rocm:$(tag)
	docker image tag transformers-rocm:$(tag) hardandheavy/transformers-rocm:latest
	docker push hardandheavy/transformers-rocm:latest

bash-dev:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		transformers-rocm:$(tag) bash

bash:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		hardandheavy/transformers-rocm:latest bash

test:
	python test/llama2.py
	python test/llama3.py

.PHONY: test
