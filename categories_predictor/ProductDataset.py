# from torch.utils.data import Dataset
# import torch

# class ProductDataset(Dataset):
#     def __init__(self, X, y, VOCAB_SIZE):
#         self.VOCAB_SIZE = VOCAB_SIZE
#         self.product_name_indices = X['product_name_indices'].values
#         self.ingredients_indices = X['ingredients_indices'].values
#         self.labels = y.values

#         # Vérification que tous les indices sont dans les limites du vocabulaire
#         for product_name, ingredients in zip(self.product_name_indices, self.ingredients_indices):
#             for idx in ingredients:
#                 if idx >= self.VOCAB_SIZE or idx < 0:
#                     print(f"Indice hors limite trouvé : {idx}")
#                     print(f"Vocabulaire max (VOCAB_SIZE): {self.VOCAB_SIZE}")
#                     print(f"Indices des ingrédients : {ingredients}")
#                     # replace out of vocab indices with 0
#                     ingredients = [0 if idx >= self.VOCAB_SIZE or idx < 0 else idx for idx in ingredients]

#     def __len__(self):
#         return len(self.product_name_indices)

#     def __getitem__(self, idx):
#         product_name = torch.tensor(self.product_name_indices[idx], dtype=torch.long)
#         ingredients = torch.tensor(self.ingredients_indices[idx], dtype=torch.long)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)

#         return {
#             'product_name': product_name,
#             'ingredients': ingredients,
#             'label': label
#         }


from torch.utils.data import Dataset
import torch

class ProductDataset(Dataset):
    def __init__(self, X, y, VOCAB_SIZE):
        self.VOCAB_SIZE = VOCAB_SIZE
        self.product_name_indices = X['product_name_indices'].values
        self.ingredients_indices = X['ingredients_indices'].values
        self.category_indices = X['category_indices'].values
        self.labels = y.values

        for product_name, ingredients, categories in zip(self.product_name_indices, self.ingredients_indices, self.category_indices):
            for idx in ingredients:
                if idx >= self.VOCAB_SIZE or idx < 0:
                    print(f"Indice hors limite trouvé : {idx}")
                    print(f"Vocabulaire max (VOCAB_SIZE): {self.VOCAB_SIZE}")
                    print(f"Indices des ingrédients : {ingredients}")
                    ingredients = [0 if idx >= self.VOCAB_SIZE or idx < 0 else idx for idx in ingredients]

            for idx in categories:
                if idx >= self.VOCAB_SIZE or idx < 0:
                    print(f"Indice hors limite trouvé : {idx}")
                    print(f"Vocabulaire max (VOCAB_SIZE): {self.VOCAB_SIZE}")
                    print(f"Indices des catégories : {categories}")
                    categories = [0 if idx >= self.VOCAB_SIZE or idx < 0 else idx for idx in categories]

    def __len__(self):
        return len(self.product_name_indices)

    def __getitem__(self, idx):
        product_name = torch.tensor(self.product_name_indices[idx], dtype=torch.long)
        ingredients = torch.tensor(self.ingredients_indices[idx], dtype=torch.long)
        category = torch.tensor(self.category_indices[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'product_name': product_name,
            'ingredients': ingredients,
            'category': category,
            'label': label
        }