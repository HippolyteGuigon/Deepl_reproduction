FROM python:3.9

# Copier les fichiers requis dans le conteneur
COPY requirements.txt /app/requirements.txt
COPY Deepl_reproduction /app/Deepl_reproduction

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances Python
RUN pip install -r requirements.txt

# Ajouter le répertoire parent de Deepl_reproduction au PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Copier le code source du projet dans le conteneur
COPY . /app