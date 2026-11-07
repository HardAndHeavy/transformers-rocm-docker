DOCKER ?= docker
IMAGE ?= transformers-rocm
PUBLISH_IMAGE ?= hardandheavy/transformers-rocm
tag ?= 3.0.1
LOCAL_IMAGE = $(IMAGE):$(tag)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_REVISION ?= $(shell git rev-parse HEAD)

# Lock generation must be reproducible too, so the generator is pinned here and
# verified by `make lock` instead of silently using whatever uv is on PATH.
UV ?= uv
UV_VERSION ?= 0.13.7
UV_LOCK_FLAGS = --python-version 3.14 \
	--python-platform x86_64-unknown-linux-gnu \
	--prerelease disallow \
	--generate-hashes \
	--no-annotate \
	--emit-index-url \
	--custom-compile-command 'make lock'

# A multi-gigabyte layer over a home connection gets reset mid-upload often
# enough that a single push attempt is not a reliable release step. Each rerun
# reuses the layers the registry already accepted, so plain retries converge.
PUSH_RETRIES ?= 8
PUSH_RETRY_DELAY ?= 20

# $(call push_with_retries,<image reference>)
define push_with_retries
	@attempt=1; \
	while [ $$attempt -le $(PUSH_RETRIES) ]; do \
		echo "Push attempt $$attempt/$(PUSH_RETRIES): $(1)"; \
		if $(DOCKER) push $(1); then \
			break; \
		fi; \
		if [ $$attempt -eq $(PUSH_RETRIES) ]; then \
			echo "Giving up after $(PUSH_RETRIES) attempts; already-uploaded layers are kept, rerun \`make publish\` to continue."; \
			exit 1; \
		fi; \
		echo "Interrupted; retrying in $(PUSH_RETRY_DELAY)s (completed layers are reused)."; \
		sleep $(PUSH_RETRY_DELAY); \
		attempt=$$((attempt + 1)); \
	done
endef

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
		-f Dockerfile .

check:
	$(DOCKER) build --check -f Dockerfile .
	python3 -B tools/versions.py check
	python3 -B -m py_compile test/lfm3.py test/rocm_smoke.py

# Propagate versions.json into the requirements inputs, the Dockerfile
# assertion block and the README table. Run after editing versions.json.
versions:
	python3 -B tools/versions.py sync

# Run only for an intentional dependency update; normal builds use the locks.
lock:
	@test "$$($(UV) --version | cut -d' ' -f2)" = "$(UV_VERSION)" \
		|| { echo "uv $(UV_VERSION) is required, found $$($(UV) --version)"; exit 1; }
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
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) python test/lfm3.py --test

publish:
	@if [ "$(tag)" = "latest" ]; then \
		echo "Refusing to publish without an explicit release tag (for example, tag=3.0.1)."; \
		exit 1; \
	fi
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):$(tag)
	$(call push_with_retries,$(PUBLISH_IMAGE):$(tag))
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):latest
	$(call push_with_retries,$(PUBLISH_IMAGE):latest)

.PHONY: build check versions lock bash smoke-test test publish
