# Utiliser l'image officielle de MySQL
FROM mysql:latest

# Définir un argument pour le mot de passe root
#ARG ROOT_PASSWORD
#ARG DATABASE
#ARG USER 
#ARG PASSWORD 
# Définir les variables d'environnement pour la base de données
ENV MYSQL_ROOT_PASSWORD=Bisma4And7
ENV MYSQL_DATABASE=deepl_database 
ENV MYSQL_USER=Hippolyte
ENV MYSQL_PASSWORD=mogalys900

# Exposer le port MySQL
EXPOSE 3306
