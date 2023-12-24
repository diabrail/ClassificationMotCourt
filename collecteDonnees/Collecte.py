import requests

class DataDownloader:
    def __init__(self, url, local_filename):
        self.url = url
        self.local_filename = local_filename

    def download_data(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            with open(self.local_filename, 'wb') as file:
                file.write(response.content)
            print(f"Les données ont été téléchargées avec succès dans le fichier {self.local_filename}.")
        else:
            print(f"Échec du téléchargement. Statut de la requête : {response.status_code}")