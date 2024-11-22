from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict, load_dataset
import torch
import os
import librosa
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# Cargar y preprocesar los datos
train_dataset_path = 'D:/Asitec/src/esp/updated_train.tsv'  # Ruta a train.tsv
valid_dataset_path = 'D:/Asitec/src/esp/updated_validated.tsv'  # Ruta a validated.tsv

if os.path.exists(train_dataset_path):
    print("El archivo existe.")
else:
    print("El archivo no se encuentra en la ruta especificada.")
normalized_path = os.path.normpath(train_dataset_path)
print("Ruta normalizada:", normalized_path)

def load_common_voice_dataset(filepath):
    """Carga los datos de Common Voice desde un archivo TSV."""
    df = pd.read_csv(filepath, delimiter='\t')
    return Dataset.from_pandas(df)

train_df = load_common_voice_dataset(train_dataset_path)
valid_df = load_common_voice_dataset(valid_dataset_path)

# Cargar procesador (tokenizador y extractor de características)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")

# Preprocesamiento de audio
def preprocess_function(examples):
    audio_path = examples["path"]
    sentence = examples["sentence"]
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)  # 16kHz para Wav2Vec 2.0
    input_values = processor(audio_array, return_tensors="pt", sampling_rate=16000).input_values
    # Convertir la transcripción en un tensor de etiquetas, mapeando a IDs del vocabulario
    labels = processor.tokenizer(sentence, padding="max_length", truncation=True, max_length=128).input_ids
    return {"input_values": input_values.squeeze(0), "labels": torch.tensor(labels)}

# Aplicamos el preprocesamiento a ambos datasets
train_dataset = train_df.map(preprocess_function, remove_columns=["client_id", "path", "sentence"])
valid_dataset = valid_df.map(preprocess_function, remove_columns=["client_id", "path", "sentence"])

# Cargar modelo preentrenado
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")

# Función para hacer el padding manualmente
def collate_fn(batch):
    # Asegurarse de que cada entrada sea un tensor antes de aplicar pad_sequence
    input_values = pad_sequence([torch.tensor(item["input_values"]) for item in batch], batch_first=True, padding_value=0.0)
    labels = pad_sequence([torch.tensor(item["labels"]) for item in batch], batch_first=True, padding_value=-100)  # -100 es el valor para el padding en CTC
    return {"input_values": input_values, "labels": labels}

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./wav2vec2_model",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    save_steps=500,
    save_total_limit=2,
)

# Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor.feature_extractor,  # Usar el feature_extractor del procesador
    data_collator=collate_fn,  # Usar el collator adecuado
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el procesador
model.save_pretrained("./voz_tex_model")
processor.save_pretrained("./voz_tex_processor")

print("Modelo y procesador guardados.")
