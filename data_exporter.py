import pymongo
import csv
from datetime import datetime

# Connexion à MongoDB
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["uptake"]
collection = db["uptakes"]

# Génération du nom de fichier avec la date et l'heure actuelles
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"uptakes_{timestamp}.csv"

# Ouverture du fichier CSV en mode écriture
with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    # Création du writer CSV
    writer = csv.writer(csvfile)
    
    # Écriture des en-têtes
    writer.writerow(["familyId", "lat", "lon", "composition", "ean", "dateTime", "quantity", "mouvementType"])
    
    # Récupération et écriture des données
    for doc in collection.find():
        writer.writerow([
            doc["family"]["familyId"],
            doc["family"]["coords"]["lat"],
            doc["family"]["coords"]["lon"],
            doc["family"]["composition"],
            doc["ean"],
            doc["dateTime"],
            doc["quantity"],
            doc["mouvementType"]
        ])

print(f"Exportation terminée. Fichier CSV sauvegardé sous : {filename}")

# Fermeture de la connexion MongoDB
client.close()