# Projet de Classification de Produits Alimentaires

## 1. Données d'entrainement

Pour les données d'entrainement, nous avons utilisé un jeu de données de produits alimentaires. Ce jeu de données est disponible sur le site de l'Open Food Facts (https://world.openfoodfacts.org/).
Malheureusement, les données sont incomplètes et les produits ne sont pas toujours bien décrits. Ils n'ont pas de catégorie définie pour les produits.
Pour les produits qui ont une catégorie, elles sont définies comme suit :

```
categories_old: "Snacks, Sweet snacks, Confectioneries, Chewing gum, Sugar-free chewing…"
OU
categories: "Snacks, Sweet snacks, Confectioneries, Chewing gum, Sugar-free chewing…"
```

L'exemple que j'ai pris était pour un produit 'Chocolat au lait' et déjà là on peut voir que la catégorie est mal définie (Chewing gum ???).

Pour cela, l'API de Chat GPT a été utilisée pour obtenir un jeu de données avec des catégories.
Voici les catégories possibles :
```
{
    'Alcool',
    'Boissons',
    'Boissons sucrées',
    'Condiments et sauces',
    'Eau',
    'Fruits et légumes',
    'Féculents',
    'Féculents et légumes secs (légumineuses)',
    'Matières grasses',
    'Plats préparés',
    'Produits laitiers',
    'Produits sucrés',
    'Sel',
    'UNDEFINED',
    'Viande, poisson et fruits de mer',
    'Vitamines et minéraux'
}
```

Pour obtenir un jeu de données avec des catégories, l'API de Chat GPT a été utilisée. Une liste de produits, comprenant leurs noms et ingrédients, lui a été fournie, et l'API devait en déduire la catégorie correspondante.

```json
{
    "product_name": "Chocolat",
    "category": "Produits sucrés",
    "code_bar": "0000018568011"
},
{
    "product_name": "Tripoux",
    "category": "Plats préparés",
    "code_bar": "0000018927207"
},
{
    "product_name": "Côtes du Rhône Villages 2014",
    "category": "Alcool",
    "code_bar": "0000020004552"
}
```

Ici il y a eu un oubli de ma part, les produits que j'ai donnés à l'API n'étaient pas mélangés et donc les catégories sont les mêmes pour les produits similaires, et dans le jeu de données d'entrainement, ce qui a fait que certains types de produits ont été que très peu représentés dans le jeu de données d'entrainement. De plus il faut garder en tête que même Chat GPT fait des erreurs dans la classification des produits.

Maintenant que nous avons un jeu de données avec des catégories, nous pouvons commencer à préparer les données pour l'entrainement du modèle.

## 2. Tokenisation

La première étape consiste à prendre le texte brut (comme un nom de produit ou la liste des ingrédients) et à le tokeniser, c'est-à-dire le diviser en une liste de mots individuels.
De plus il faut enlever les mots vides (stop words) qui sont des mots qui n'ont pas de sens et qui n'apportent pas d'information utile pour la classification des produits. Par exemple, les mots "le", "la", "de", "du", "des", etc. Cela permet de réduire la taille du vocabulaire, et donc d'avoir un plus petit modèle.

Exemple : Pour un produit comme "Pain au chocolat", la tokenisation pourrait donner une liste comme :
```
["Pain", "chocolat"]
```

De même, pour les ingrédients "farine, beurre, chocolat", la tokenisation donnerait :
```
["farine", "beurre", "chocolat"]
```

Pour cette étape, nous allons utiliser la fonction `tokenize_and_remove_stopwords`.

## 3. Vectorisation

Pour cela, il faut premièrement créer un vocabulaire à partir des mots des produits, des ingrédients et des catégories. Cette étape consiste à rassembler tous les tokens uniques présents dans ces trois colonnes de données, de manière à obtenir un dictionnaire qui associe chaque mot à un index unique. Ces index seront ensuite utilisés pour représenter chaque élément sous forme de vecteurs numériques.

Après cela on a donc une liste de mots uniques, on peut maintenant transformer chaque produit en un vecteur numérique. Pour cela, on va utiliser la fonction `tokens_to_indices`.

Avant de vectoriser les données, certains produits contiennent plus de 220 mots, ce qui risquerait de surcharger le modèle avec des informations inutiles. Pour éviter ce problème, il est important d'analyser la distribution de la longueur des listes de tokens (mots) afin de définir des limites raisonnables.

## GRAPHIQUE DE LA DISTRIBUTION DE LA LONGUEUR DES PRODUITS

En se basant sur ces distributions, on peut fixer les longueurs maximales pour chaque type de donnée :
```	
MAX_LEN_PRODUCT_NAME = 7
MAX_LEN_INGREDIENTS = 60
MAX_LEN_CATEGORY = 11
```

Ensuite, les données peuvent être vectorisées en utilisant la fonction tokens_to_indices. Pour chaque liste de mots, cette fonction retourne un vecteur où chaque mot est remplacé par son index dans le vocabulaire. La taille de ce vecteur est limitée à une longueur maximale, définie en fonction de la distribution des données. Si la liste dépasse cette longueur, elle est tronquée ; si elle est plus courte, elle est complétée avec des zéros.

Par exemple: Pain au chocolat -> [1, 2, 0, 0, 0, 0, 0] (si "Pain" a pour index 1 et "chocolat" a pour index 2)

## 4. Création du modèle

Avant de donner les valeurs au modèle, on applique un embedding sur les données (c'est une étape de prétraitement, distinct du réseau de neurones lui-même). Pour chaque mot (représenté par un index dans le vocabulaire), l'embedding transforme cet index en un vecteur de X dimensions. Ce vecteur capture les informations sémantiques du mot, ce qui permet au modèle de mieux comprendre les relations entre les mots.

Par exemple :
Si on prend les mots "gateau" et "pomme", ils sont d'abord convertis en indices (par exemple, 125 pour "gateau" et 12 pour "pomme").
On a donc [125, 12] comme entrée pour l'embedding.

Ensuite, avec un embedding de 5 dimensions, "125" devient quelque chose comme [0.25, -0.43, 0.67, -0.12, 0.58] et "12" devient [0.41, 0.21, -0.37, 0.45, -0.19].
```
[
    [0.25, -0.43, 0.67, -0.12, 0.58],
    [0.41, 0.21, -0.37, 0.45, -0.19]
]
```

Cela permet d'avoir une représentation vectorielle (graphique) qui capture les relations entre les mots.

La couche LSTM est utilisée pour traiter des informations séquentielles, où chaque élément donné à cette couche influence les éléments suivants. À chaque étape N, la sortie dépend non seulement de l'élément actuel N, mais aussi de l'élément précédent N−1. Cela permet à la couche de "se souvenir" de ce qui s'est passé avant dans la séquence. Grâce à cela, l'LSTM peut mieux comprendre les relations entre les mots dans une phrase, ce qui aide à améliorer la classification des produits.

Contrairement à la couche dense, où chaque élément est traité indépendamment des autres, la couche LSTM garde une mémoire des informations passées au sein de la séquence. Elle conserve en mémoire uniquement ce qu'elle a vu dans la séquence actuelle (par exemple, les mots précédents dans une phrase), et non tout ce qu'elle a vu lors de l'entraînement. Cette mémoire est gérée étape par étape, au fur et à mesure de la séquence, ce qui permet à l'LSTM de savoir quelles informations réutiliser ou oublier en fonction du contexte.

Enfin il passe par une couche dense pour la classification des produits. Où il donnera la probabilité que le produit appartient à une catégorie donnée.
