STACK_NAME ?= network


init:
	docker network create ingress | true
	COMPOSE_PROJECT_NAME=init docker compose -f init.yaml --env-file .env up -d --remove-orphans

down:
	COMPOSE_PROJECT_NAME=init docker compose -f init.yaml down
	COMPOSE_PROJECT_NAME=init docker compose -f init.yaml rm
	docker network rm ingress

service-build:
	COMPOSE_PROJECT_NAME=$(STACK_NAME) docker compose -f $(STACK_NAME).yaml --env-file .env build

service-up:
	COMPOSE_PROJECT_NAME=$(STACK_NAME) docker compose -f $(STACK_NAME).yaml --env-file .env up -d --remove-orphans

service-restart:
	COMPOSE_PROJECT_NAME=$(STACK_NAME) docker compose -f $(STACK_NAME).yaml --env-file .env restart

service-logs:
	COMPOSE_PROJECT_NAME=$(STACK_NAME) docker compose -f $(STACK_NAME).yaml --env-file .env logs

service-down:
	COMPOSE_PROJECT_NAME=$(STACK_NAME) docker compose -f $(STACK_NAME).yaml --env-file .env down
	COMPOSE_PROJECT_NAME=$(STACK_NAME) docker compose -f $(STACK_NAME).yaml --env-file .env rm

openclaw-cli:
	COMPOSE_PROJECT_NAME=openclaw docker compose -f openclaw.yaml --profile cli --env-file .env run --rm openclaw-cli $(ARGS)

restore-nocodb:
	# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION are needed.
	litestream restore -o test-backup.db s3://roundtable-nocodb