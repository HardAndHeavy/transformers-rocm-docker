DOCKER ?= docker
IMAGE ?= transformers-rocm
PUBLISH_IMAGE ?= hardandheavy/transformers-rocm
tag ?= 3.0.0
LOCAL_IMAGE = $(IMAGE):$(tag)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_REVISION ?= $(shell git rev-parse HEAD)
UV ?= uv
UV_LOCK_FLAGS = --python-version 3.14 \
	--python-platform x86_64-unknown-linux-gnu \
	--prerelease disallow \
	--generate-hashes \
	--no-annotate \
	--emit-index-url \
	--custom-compile-command 'make lock'

DOCKER_RUN_FLAGS = --rm \
	--ipc=host \
	--device=/dev/kfd \
	--device=/dev/dri \
	-w /app \
	-v $(CURDIR):/app \
	-v $(CURDIR)/huggingface:/root/.cache/huggingface

build:
	$(DOCKER) build \
		--build-arg IMAGE_VERSION=$(tag) \
		--build-arg IMAGE_REVISION=$(GIT_REVISION) \
		--build-arg IMAGE_CREATED=$(BUILD_DATE) \
		-t $(LOCAL_IMAGE) \
		-t $(IMAGE):latest \
		-f Dockerfile .

check:
	$(DOCKER) build --check -f Dockerfile .
	python3 -m py_compile test/lfm3.py test/rocm_smoke.py

# Run only for an intentional dependency update; normal builds use the locks.
lock:
	$(UV) pip compile requirements-bootstrap.in \
		--output-file requirements-bootstrap.lock \
		$(UV_LOCK_FLAGS)
	$(UV) pip compile requirements.in \
		--output-file requirements.lock \
		--index https://stable.repo.amd.com/rocm/whl-next/ \
		--default-index https://pypi.org/simple \
		--index-strategy first-index \
		--no-strip-extras \
		$(UV_LOCK_FLAGS)

bash:
	$(DOCKER) run -it $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) bash

smoke-test:
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) python test/rocm_smoke.py

test:
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) python test/lfm3.py --auto-test

publish:
	@if [ "$(tag)" = "latest" ]; then \
		echo "Refusing to publish without an explicit release tag (for example, tag=3.0.0)."; \
		exit 1; \
	fi
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):$(tag)
	$(DOCKER) push $(PUBLISH_IMAGE):$(tag)
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):latest
	$(DOCKER) push $(PUBLISH_IMAGE):latest

.PHONY: test
