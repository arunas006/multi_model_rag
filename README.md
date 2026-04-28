python -m ingestion.loader
uv run python -m ingestion.pipeline
docker compose -f docker-compose-parser.yml logs -f
docker ps
docker compose -f docker-compose-parser.yml down
docker compose -f docker-compose-parser.yml up --build