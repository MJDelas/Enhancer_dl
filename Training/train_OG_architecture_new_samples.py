import numpy as np
from Bio import SeqIO
import tensorflow as tf
import pandas as pd
import keras
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import pearsonr
import os
import argparse
import wandb
from wandb.integration.keras import WandbCallback

def parse_args():
    parser = argparse.ArgumentParser(description='Train your model with different learning rate options.')
    parser.add_argument('--name', type=str, required=True, help='Base name for the files and suffix for the predictions file.')
    parser.add_argument('--label_column', type=str, default='mean_count')
    parser.add_argument('--lr_option', type=str, default="0.001", help='Learning rate for training. Default is 0.001.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--train_sequences_path', type=str, required=True, help='Path to training sequences file.')
    parser.add_argument('--train_labels_path', type=str, required=True, help='Path to training labels file.')
    parser.add_argument('--val_sequences_path', type=str, required=True, help='Path to validation sequences file.')
    parser.add_argument('--val_labels_path', type=str, required=True, help='Path to validation labels file.')
    parser.add_argument('--test_sequences_path', type=str, required=True, help='Path to test sequences file.')
    parser.add_argument('--test_labels_path', type=str, required=True, help='Path to test labels file.')
    return parser.parse_args()

def main():
    args = parse_args()
    name = args.name
    lr_option = args.lr_option
    label_column = args.label_column
    epochs = args.epochs
    batch_size = args.batch_size
    train_sequences_path = args.train_sequences_path
    train_labels_path = args.train_labels_path
    val_sequences_path = args.val_sequences_path
    val_labels_path = args.val_labels_path
    test_sequences_path = args.test_sequences_path
    test_labels_path = args.test_labels_path

    wandb.init(project="OG Architecture New", job_type=name, config={
        "learning_rate": lr_option,
        "epochs": epochs,
        "batch_size": batch_size
    })
    
    train_model(name, lr_option, label_column, epochs, batch_size,
                train_sequences_path, train_labels_path, val_sequences_path, val_labels_path,
                test_sequences_path, test_labels_path)

def train_model(name, lr_option, label_column, epochs, batch_size,
                train_sequences_path, train_labels_path, val_sequences_path, val_labels_path,
                test_sequences_path, test_labels_path):
    max_length = 1001  # Adjust based on your sequences' lengths

    train_sequences, train_labels = load_data(train_sequences_path, train_labels_path, max_length, label_column)
    val_sequences, val_labels = load_data(val_sequences_path, val_labels_path, max_length, label_column)
    test_sequences, test_labels = load_data(test_sequences_path, test_labels_path, max_length, label_column)

    input_shape = (max_length, 4)
    model = build_cnn(input_shape)
    learning_rate = float(lr_option)  # Convert lr_option to a float directly
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mse', pearson_correlation])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        train_sequences, train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, WandbCallback()],
        shuffle=True
    )

    predictions = model.predict(test_sequences)
    predictions_file_path = f'models/prediction_{name}_{lr_option}.csv'
    test_labels_df = pd.read_csv(test_labels_path, usecols=[label_column])
    predictions_df = pd.DataFrame({'True Label': test_labels_df[label_column], 'Prediction': predictions.flatten()})
    predictions_df.to_csv(predictions_file_path, index=False)

def load_data(sequences_path, labels_path, max_length, label_column):
    sequences = list(SeqIO.parse(sequences_path, "fasta"))
    labels = pd.read_csv(labels_path, usecols=[label_column])[label_column].values
    
    # One-hot encode sequences
    encoded_sequences = np.array([one_hot_encode_sequence(str(seq.seq), max_length) for seq in sequences], dtype=np.float32)
    labels = labels.reshape(-1, 1).astype(np.float32)
    
    # Shuffle data
    indices = np.arange(len(encoded_sequences))
    np.random.shuffle(indices)
    encoded_sequences = encoded_sequences[indices]
    labels = labels[indices]
    
    return encoded_sequences, labels

def one_hot_encode_sequence(sequence, max_length):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1]}
    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    padding = [[0, 0, 0, 0]] * (max_length - len(encoded))
    return np.array(encoded + padding, dtype=np.float32)

def pearson_correlation(y_true, y_pred):
    x = y_true - tf.reduce_mean(y_true)
    y = y_pred - tf.reduce_mean(y_pred)
    correlation = tf.reduce_sum(x * y) / (tf.sqrt(tf.reduce_sum(tf.square(x))) * tf.sqrt(tf.reduce_sum(tf.square(y))))
    return correlation

def build_cnn(input_shape):
    input_layer = Input(shape=input_shape, name='input_3')

    conv1 = Conv1D(filters=256, kernel_size=7, padding='same', activation='linear', name='Conv1D_1st')(input_layer)
    bn1 = BatchNormalization(axis=-1, name='batch_normalization_12')(conv1)
    act1 = Activation('relu', name='activation_12')(bn1)
    pool1 = MaxPooling1D(pool_size=3, strides=3, padding='valid', name='max_pooling1d_8')(act1)

    conv2 = Conv1D(filters=120, kernel_size=3, padding='same', activation='linear', name='Conv1D_2')(pool1)
    bn2 = BatchNormalization(axis=-1, name='batch_normalization_13')(conv2)
    act2 = Activation('relu', name='activation_13')(bn2)
    pool2 = MaxPooling1D(pool_size=3, strides=3, padding='valid', name='max_pooling1d_9')(act2)

    conv3 = Conv1D(filters=60, kernel_size=3, padding='same', activation='linear', name='Conv1D_3')(pool2)
    bn3 = BatchNormalization(axis=-1, name='batch_normalization_14')(conv3)
    act3 = Activation('relu', name='activation_14')(bn3)
    pool3 = MaxPooling1D(pool_size=3, strides=3, padding='valid', name='max_pooling1d_10')(act3)

    conv4 = Conv1D(filters=60, kernel_size=3, padding='same', activation='linear', name='Conv1D_4')(pool3)
    bn4 = BatchNormalization(axis=-1, name='batch_normalization_15')(conv4)
    act4 = Activation('relu', name='activation_15')(bn4)
    pool4 = MaxPooling1D(pool_size=3, strides=3, padding='valid', name='max_pooling1d_11')(act4)

    flatten = Flatten(name='flatten_2')(pool4)

    dense1 = Dense(units=64, activation='linear', name='Dense_1')(flatten)
    bn5 = BatchNormalization(axis=-1, name='batch_normalization_16')(dense1)
    act5 = Activation('relu', name='activation_16')(bn5)
    dropout1 = Dropout(0.4, name='dropout_4')(act5)
    dense2 = Dense(units=256, activation='linear', name='Dense_2')(dropout1)
    bn6 = BatchNormalization(axis=-1, name='batch_normalization_17')(dense2)
    act6 = Activation('relu', name='activation_17')(bn6)
    dropout2 = Dropout(0.4, name='dropout_5')(act6)

    output_layer = Dense(units=1, activation='linear', name='Dense_output')(dropout2)
    model = Model(inputs=input_layer, outputs=output_layer, name='model_2')
    return model

if __name__ == "__main__":
    main()
