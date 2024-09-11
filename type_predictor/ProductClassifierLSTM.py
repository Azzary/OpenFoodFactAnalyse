import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, max_len_product_name, max_len_ingredients, max_len_category_indices):
        super(ProductClassifierLSTM, self).__init__()
        
        self.product_name_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.ingredients_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.product_name_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.ingredients_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.category_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 3, num_classes)  # *3 car on combine les trois LSTM (noms, ingrédients, catégories)
        
    def forward(self, product_name, ingredients, category_indices):
        product_name_embedded = self.product_name_embedding(product_name)
        _, (product_name_hidden, _) = self.product_name_lstm(product_name_embedded)
        product_name_hidden = product_name_hidden[-1]
        
        ingredients_embedded = self.ingredients_embedding(ingredients)
        _, (ingredients_hidden, _) = self.ingredients_lstm(ingredients_embedded)
        ingredients_hidden = ingredients_hidden[-1]
        
        category_embedded = self.category_embedding(category_indices)
        _, (category_hidden, _) = self.category_lstm(category_embedded)
        category_hidden = category_hidden[-1]
        
        combined = torch.cat((product_name_hidden, ingredients_hidden, category_hidden), dim=1)
        output = self.fc(combined)
        
        return output

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ProductClassifierLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, max_len_product_name, max_len_ingredients):
#         super(ProductClassifierLSTM, self).__init__()
        
#         self.product_name_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.ingredients_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
#         # LSTM pour capturer les séquences de mots dans les noms de produits
#         self.product_name_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
#         # LSTM pour capturer les séquences de mots dans les ingrédients
#         self.ingredients_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
#         # Fully connected layer pour la classification
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 car on combine les deux LSTM
        
#     def forward(self, product_name, ingredients):
#         # Embedding pour le nom du produit
#         product_name_embedded = self.product_name_embedding(product_name)
#         # Passer les embeddings à travers un LSTM
#         _, (product_name_hidden, _) = self.product_name_lstm(product_name_embedded)
#         product_name_hidden = product_name_hidden[-1]  # On prend la dernière sortie cachée du LSTM
        
#         # Embedding pour les ingrédients
#         ingredients_embedded = self.ingredients_embedding(ingredients)
#         # Passer les embeddings à travers un LSTM
#         _, (ingredients_hidden, _) = self.ingredients_lstm(ingredients_embedded)
#         ingredients_hidden = ingredients_hidden[-1]  # On prend la dernière sortie cachée du LSTM
        
#         # Concaténer les deux sorties LSTM (noms de produits et ingrédients)
#         combined = torch.cat((product_name_hidden, ingredients_hidden), dim=1)
        
#         # Passer la sortie combinée dans la couche fully connected
#         output = self.fc(combined)
        
#         return output