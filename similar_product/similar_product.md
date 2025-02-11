# Traitement des données Open Food Facts

On utilise la base de données de Open Food Facts.

Pour chaque produit, on récupère les informations suivantes :
- id
- le nom
- les ingrédients
- les calories
- les protéines
- les carbohydrates
- les graisses

## Normalisation des données nutritionnelles

On utilise StandardScaler pour les calories, protéines, carbohydrates, graisses.
Cela permet d'avoir une valeur proche de 0.

## Traitement des ingrédients

Pour les ingrédients, on les transforme en vecteur de mots, puis on les vectorise.

Pour cela, on récupère tous les différents mots (ingrédients) possibles dans tous les produits. On a donc grâce à ça un vecteur avec par exemple 10 valeurs. (si dans tous les produits on a 10 ingrédients distincts)

On utilise TfidfVectorizer pour cette étape.

On utilise TF-IDF pour attribuer à chaque ingrédient un poids en fonction de sa fréquence d'apparition dans un produit (TF) et de la rareté de cet ingrédient dans l'ensemble des produits (IDF). Ainsi, les mots qui apparaissent dans beaucoup de produits reçoivent une pondération plus faible, tandis que les mots rares (présents dans peu de produits) ont un poids plus élevé.

## Exemple

Prenons un exemple :

**Chocolat au lait**
Ingrédients : "sucre, beurre de cacao, lait en poudre, pâte de cacao, lécithine de soja"
Valeurs nutritionnelles (pour 100g) :
- Calories : 545 kcal
- Protéines : 6.3g
- Glucides : 59.4g
- Lipides : 31.5g

### Après normalisation

- Calories : 1.2
- Protéines : -0.5
- Glucides : 0.8
- Lipides : 1.5

### Traitement des ingrédients

Pour les ingrédients, on obtient un vecteur qui représente chaque ingrédient par une valeur pondérée. Supposons qu'il y ait 10 ingrédients distincts dans l'ensemble des produits :

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Imaginons maintenant que :
- "sucre" soit à l'index 0
- "beurre de cacao" à l'index 1
- "lait en poudre" à l'index 2
- "pâte de cacao" à l'index 3
- "lécithine de soja" à l'index 4

Ce sont les ingrédients de notre produit "chocolat au lait". On obtiendrait alors :

```
[0.1, 0.4, 0.4, 0.7, 0.3, 0, 0, 0, 0, 0]
```

- Sucre (index 0) a une valeur de 0.1 parce qu'il est très courant et apparaît dans beaucoup de produits. Cela lui donne un poids plus faible.
- Beurre de cacao (index 1) et lait en poudre (index 2) sont présents dans plusieurs produits, mais moins fréquemment que le sucre, donc leurs valeurs sont légèrement plus élevées à 0.4.
- Pâte de cacao (index 3) est plus rare et reçoit une pondération encore plus élevée à 0.7.
- Lécithine de soja (index 4) est aussi moins courante, avec une valeur de 0.3.

Ce vecteur peut être représenté graphiquement, bien que dans un espace à 10 dimensions (dans cet exemple), ce qui rend la visualisation difficile.
