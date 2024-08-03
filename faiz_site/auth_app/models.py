from django.db import models
from db_connection import db


chat_collection=db['chat_data']

class Utilisateur(models.Model):
    nom = models.CharField(max_length=50)  
    mot_de_passe = models.CharField(max_length=50)

