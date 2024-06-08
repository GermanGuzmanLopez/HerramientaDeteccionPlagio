import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Configurar dispositivo para CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para leer archivos de texto
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Función para asegurarse de que los nombres de archivos tengan la extensión correcta
def ensure_txt_extension(filename):
    if not filename.endswith('.txt'):
        return filename + '.txt'
    return filename

# Cargar los datos
df = pd.read_csv('Training.csv')

# Asegurarse de que 'fuentes' sea siempre una lista
df['fuentes'] = df['fuentes'].apply(lambda x: x.split('$') if pd.notna(x) else [])

# Añadir columna para el texto sospechoso
df['texto_sospechoso'] = df['archivo'].apply(lambda x: read_text_from_file(os.path.join('textos_sospechosos/', ensure_txt_extension(x))))

# Crear una columna de etiquetas para plagio/no plagio
df['plagio'] = df['es_plagio']

# Crear una columna de etiquetas para tipos de plagio
df['tipo_plagio'] = df['tipos_de_plagio'].factorize()[0]

# Verificar la distribución de las etiquetas
print(df['plagio'].value_counts())
print(df['tipo_plagio'].value_counts())

# Manejo del balance de clases para la detección de plagio
df_plagio = df[df['plagio'] == 1]
df_no_plagio = df[df['plagio'] == 0]

df_no_plagio_oversampled = df_no_plagio.sample(len(df_plagio), replace=True, random_state=42)
df_balanced = pd.concat([df_plagio, df_no_plagio_oversampled], axis=0).reset_index(drop=True)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_df, temp_df = train_test_split(df_balanced, test_size=0.3, random_state=42, stratify=df_balanced['plagio'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['plagio'])

# Tokenización
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class PlagiarismDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        if input_ids.size(0) == 0 or attention_mask.size(0) == 0:
            raise ValueError(f"Empty input_ids or attention_mask found for item {item}")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Verificación de que los datos no estén vacíos
def check_dataset(dataset):
    if len(dataset) == 0:
        raise ValueError("El dataset está vacío. Asegúrese de que los datos se han cargado y procesado correctamente.")
    for item in dataset:
        if len(item['input_ids']) == 0 or len(item['attention_mask']) == 0:
            raise ValueError("Se encontró un ejemplo vacío en el dataset. Revise el procesamiento de datos.")

# Crear datasets
train_dataset = PlagiarismDataset(
    texts=train_df['texto_sospechoso'].tolist(),
    labels=train_df['plagio'].tolist(),
    tokenizer=tokenizer
)

val_dataset = PlagiarismDataset(
    texts=val_df['texto_sospechoso'].tolist(),
    labels=val_df['plagio'].tolist(),
    tokenizer=tokenizer
)

test_dataset = PlagiarismDataset(
    texts=test_df['texto_sospechoso'].tolist(),
    labels=test_df['plagio'].tolist(),
    tokenizer=tokenizer
)

# Verificar datasets
check_dataset(train_dataset)
check_dataset(val_dataset)
check_dataset(test_dataset)

# Configurar el modelo y los argumentos de entrenamiento con dropout
class BertForSequenceClassificationWithDropout(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassificationWithDropout, self).__init__(config)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        logits = outputs[1]
        logits = self.dropout(logits)
        return (outputs[0], logits)

model = BertForSequenceClassificationWithDropout.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: classification_report(p.label_ids, p.predictions.argmax(-1), output_dict=True)
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
trainer.evaluate()

# Guardar el modelo
model.save_pretrained('./plagiarism-model')
tokenizer.save_pretrained('./plagiarism-tokenizer')

# Evaluar en el conjunto de prueba
predictions, labels, _ = trainer.predict(test_dataset)
preds = predictions.argmax(-1)

# Reporte de clasificación para plagio/no plagio
print("Reporte de clasificación para plagio/no plagio:")
print(classification_report(labels, preds, target_names=['No Plagio', 'Plagio']))

# Manejo de tipos de plagio

# Filtrar clases con menos de 2 muestras
df_tipo_plagio = df[df['tipo_plagio'].isin(df['tipo_plagio'].value_counts()[df['tipo_plagio'].value_counts() >= 2].index)]

# Diccionario para mapear números de etiquetas a nombres de categorías
label_dict = {i: label for i, label in enumerate(df['tipos_de_plagio'].unique())}

# Dividir los datos en conjuntos de entrenamiento, validación y prueba para tipos de plagio
train_df_tipo, temp_df_tipo = train_test_split(df_tipo_plagio, test_size=0.3, random_state=42, stratify=df_tipo_plagio['tipo_plagio'])
val_df_tipo, test_df_tipo = train_test_split(temp_df_tipo, test_size=0.5, random_state=42, stratify=temp_df_tipo['tipo_plagio'])

# Crear datasets para tipos de plagio
train_dataset_tipo = PlagiarismDataset(
    texts=train_df_tipo['texto_sospechoso'].tolist(),
    labels=train_df_tipo['tipo_plagio'].tolist(),
    tokenizer=tokenizer
)

val_dataset_tipo = PlagiarismDataset(
    texts=val_df_tipo['texto_sospechoso'].tolist(),
    labels=val_df_tipo['tipo_plagio'].tolist(),
    tokenizer=tokenizer
)

test_dataset_tipo = PlagiarismDataset(
    texts=test_df_tipo['texto_sospechoso'].tolist(),
    labels=test_df_tipo['tipo_plagio'].tolist(),
    tokenizer=tokenizer
)

# Verificar datasets para tipos de plagio
check_dataset(train_dataset_tipo)
check_dataset(val_dataset_tipo)
check_dataset(test_dataset_tipo)

# Configurar el modelo y los argumentos de entrenamiento para tipos de plagio
model_tipo = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['tipo_plagio'].unique()))
model_tipo.to(device)

training_args_tipo = TrainingArguments(
    output_dir='./results_tipo_plagio',
    num_train_epochs=10,  # Aumentar el número de épocas
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_tipo_plagio',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    learning_rate=5e-5,  # Aumentar la tasa de aprendizaje
)

trainer_tipo = Trainer(
    model=model_tipo,
    args=training_args_tipo,
    train_dataset=train_dataset_tipo,
    eval_dataset=val_dataset_tipo,
    compute_metrics=lambda p: classification_report(p.label_ids, p.predictions.argmax(-1), output_dict=True, zero_division=0)
)

# Entrenar el modelo para tipos de plagio
trainer_tipo.train()

# Evaluar el modelo para tipos de plagio
trainer_tipo.evaluate()

# Guardar el modelo para tipos de plagio
model_tipo.save_pretrained('./plagiarism_tipo_model')
tokenizer.save_pretrained('./plagiarism_tipo_tokenizer')

# Evaluar en el conjunto de prueba para tipos de plagio
predictions_tipo, labels_tipo, _ = trainer_tipo.predict(test_dataset_tipo)
preds_tipo = predictions_tipo.argmax(-1)

# Mapeo de etiquetas numéricas a nombres de categorías
labels_named = [label_dict[label] for label in labels_tipo]
preds_named = [label_dict[pred] for pred in preds_tipo]

# Reporte de clasificación para tipos de plagio
print("Reporte de clasificación para tipos de plagio:")
print(classification_report(labels_named, preds_named, zero_division=0))

# Matriz de confusión para tipos de plagio
cm = confusion_matrix(labels_named, preds_named, labels=list(label_dict.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_dict.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión para Tipos de Plagio")
plt.show()
