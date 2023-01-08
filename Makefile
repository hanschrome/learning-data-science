help:
	@cat Makefile
bash:
	docker compose run -it python bash || docker-compose run -it python bash

