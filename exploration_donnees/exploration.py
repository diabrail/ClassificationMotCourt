import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    def __init__(self, filename):
        self.filename = filename
        self.df = None  # Ajoutez une variable pour stocker le DataFrame exploré

    def explore_data(self):
        # Charger les données dans un DataFrame
        self.df = pd.read_csv(self.filename, delimiter='\t', header=None, names=['Label', 'Text'])

        # Afficher les premières lignes du DataFrame pour comprendre la structure des données
        print(self.df.head())

        # Distribution des classes
        print("----------Distribution des classes----------------")
        class_distribution = self.df['Label'].value_counts()
        print("Distribution des classes :\n", class_distribution)

        # Longueur des tweets
        self.df['Tweet_Length'] = self.df['Text'].apply(len)


    def get_explored_data(self):
        # Renvoie le DataFrame exploré
        return self.df