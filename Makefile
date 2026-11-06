DOCKER ?= docker
IMAGE ?= transformers-rocm
TAG ?= latest
LOCAL_IMAGE = $(IMAGE):$(TAG)
PUBLISH_IMAGE ?= hardandheavy/transformers-rocm

DOCKER_RUN_FLAGS = --rm \
	--ipc=host \
	--device=/dev/kfd \
	--device=/dev/dri \
	-w /app \
	-v $(CURDIR):/app \
	-v $(CURDIR)/huggingface:/root/.cache/huggingface

build:
	$(DOCKER) build --pull -t $(LOCAL_IMAGE) -f Dockerfile .

publish: build
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):$(TAG)
	$(DOCKER) push $(PUBLISH_IMAGE):$(TAG)
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):latest
	$(DOCKER) push $(PUBLISH_IMAGE):latest

huggingface:
	mkdir -p $@

# Interactive shells and tests reuse the local image. Run `make build`
# explicitly after changing the image definition; otherwise the large ROCm
# dependency set would be resolved again before every shell or test run.
ensure-image:
	@$(DOCKER) image inspect $(LOCAL_IMAGE) >/dev/null 2>&1 || \
		$(MAKE) --no-print-directory build

bash: ensure-image huggingface
	$(DOCKER) run -it $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) bash

test: ensure-image huggingface
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) python test/lfm3.py

.PHONY: build publish ensure-image bash test
