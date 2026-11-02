# Image d'execution du prototype Fraude-360.
# Base slim + JRE : PySpark exige une JVM, rien de plus.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    FRAUDE360_HOME=/app

WORKDIR /app

# les dependances d'abord : la couche est mise en cache tant que
# requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY conf/ ./conf/
COPY tests/ ./tests/
COPY sas/ ./sas/

# utilisateur non privilegie : un conteneur qui tourne en root est un
# constat d'audit, pas un detail
RUN useradd --create-home --uid 10001 fraude && chown -R fraude:fraude /app
USER fraude

ENTRYPOINT ["python", "-m", "fraude360"]
CMD ["tout", "--config", "conf/config_rapide.yaml"]
