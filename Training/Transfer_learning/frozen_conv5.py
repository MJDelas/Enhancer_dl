import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from tensorflow.keras import Model

# Ensure TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified hyperparameters.")
    parser.add_argument('--label_column', type=str, default='WT_D5_p1_NFIAn.mRp.clN.bigWig')
    parser.add_argument('--lr_option', type=float, required=True, help='Learning rate option for training the model.')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs for training the model.')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    lr_option = args.lr_option
    label_column = args.label_column
    epochs = args.epochs

    print_pearsons_correlation(label_column)
    train_model(lr_option, label_column, epochs)

def load_pretrained_model():
    # Load model architecture from JSON
    model_path = '/camp/home/weie/lab_space_weie/almeida/groups/stark/almeida/Papers/Enhancer_design/Data/Accessibility_model_files/Results_fold08_CNS_DeepSTARR2_rep3/'
    with open(model_path + 'Model.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Load weights into the model
    model.load_weights(model_path + 'Model.h5')

    # Freeze convolutional layers
    for layer in model.layers:
        if isinstance(layer, Conv1D):
            layer.trainable = False

    # Add a new dense layer before the output layer
    # We need to recreate the model to add the layer
    x = model.layers[-2].output  # Get the output of the layer before the last
    x = Dense(units=256, activation='relu', name='Dense_3')(x)
    output = model.layers[-1](x)  # Connect the new dense layer to the original output layer

    # Create a new model with the modified architecture
    new_model = Model(inputs=model.input, outputs=output)

    return new_model



class PlotMetrics(Callback):
    def __init__(self, epochs, label_column, learning_rate):
        super(PlotMetrics, self).__init__()
        self.epochs = epochs  # Store epochs for use in callback
        self.label_column = label_column
        self.learning_rate = learning_rate

        self.fig = None
        self.logs = None
        self.metrics_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'pearson_correlation': [],
            'val_pearson_correlation': []
        }

        # Initialize the plot and save the first version
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Pearson Correlation')
        self.line1, = self.ax1.plot([], [], label='loss')
        self.line2, = self.ax1.plot([], [], label='val_loss')
        self.line3, = self.ax2.plot([], [], label='pearson_correlation')
        self.line4, = self.ax2.plot([], [], label='val_pearson_correlation')
        self.ax1.legend()
        self.ax2.legend()
        self.fig.tight_layout()

    def on_train_begin(self, logs=None):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.pearsons = []
        self.val_pearsons = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        self.logs = logs  # Store the logs for further processing

        self.x.append(self.i)
        self.losses.append(logs.get('loss', np.nan))
        self.val_losses.append(logs.get('val_loss', np.nan))
        self.pearsons.append(logs.get('pearson_correlation', np.nan))
        self.val_pearsons.append(logs.get('val_pearson_correlation', np.nan))
        self.i += 1
        
        self.line1.set_data(self.x, self.losses)
        self.line2.set_data(self.x, self.val_losses)
        self.line3.set_data(self.x, self.pearsons)
        self.line4.set_data(self.x, self.val_pearsons)
        
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Save plot for the current epoch with the specific filename
        save_path = f'figures/added_layer/{self.label_column}_{self.learning_rate}_current_epoch.png'
        self.fig.savefig(save_path)
        print(f"Saved plot for epoch {epoch + 1} at: {os.path.abspath(save_path)}")

        # Append metrics to history
        self.metrics_history['epoch'].append(epoch + 1)  # epoch is zero-indexed
        self.metrics_history['loss'].append(self.losses[-1])
        self.metrics_history['val_loss'].append(self.val_losses[-1])
        self.metrics_history['pearson_correlation'].append(self.pearsons[-1])
        self.metrics_history['val_pearson_correlation'].append(self.val_pearsons[-1])

    def save_to_csv(self, file_path):
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(file_path, index=False)
        print(f"Saved metrics history to: {os.path.abspath(file_path)}")

def print_pearsons_correlation(label_column):
    file_path_sequences_val = '/camp/home/weie/lab_space_weie/training/new_samples/val_new.fasta'
    file_path_labels_val = '/camp/home/weie/lab_space_weie/training/new_samples/val_new_dup.csv'
    max_length = 1001  # Adjust based on your sequences' lengths
    batch_size = 128
    
    val_dataset = load_combined_dataset(file_path_sequences_val, file_path_labels_val, max_length, batch_size, label_column, shuffle=False)

    model = load_pretrained_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[pearson_correlation])

    # Count samples for calculating steps
    num_val_samples = count_samples(file_path_sequences_val)
    num_val_steps = (num_val_samples + batch_size - 1) // batch_size

    # Evaluate Pearson's correlation on validation dataset
    val_loss, val_pearson = model.evaluate(val_dataset, steps=num_val_steps)
    print(f"Validation Pearson Correlation (before training): {val_loss, val_pearson}")

def train_model(learning_rate, label_column, epochs):
    model = load_pretrained_model()

    file_path_sequences_train = '/camp/home/weie/lab_space_weie/training/new_samples/train_new.fasta'
    file_path_labels_train = '/camp/home/weie/lab_space_weie/training/new_samples/train_new_dup.csv'
    file_path_sequences_val = '/camp/home/weie/lab_space_weie/training/new_samples/val_new.fasta'
    file_path_labels_val = '/camp/home/weie/lab_space_weie/training/new_samples/val_new_dup.csv'
    max_length = 1001  # Adjust based on your sequences' lengths
    batch_size = 128
    
    train_dataset = load_combined_dataset(file_path_sequences_train, file_path_labels_train, max_length, batch_size, label_column, shuffle=True)
    val_dataset = load_combined_dataset(file_path_sequences_val, file_path_labels_val, max_length, batch_size, label_column, shuffle=False)

    input_shape = (max_length, 4)

    plot_metrics = PlotMetrics(epochs,label_column,learning_rate)  # Initialize PlotMetrics with epochs
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mse', pearson_correlation])

    # Count samples for calculating steps per epoch
    num_train_samples = count_samples(file_path_sequences_train)
    num_val_samples = count_samples(file_path_sequences_val)
    print(f"Number of training samples: {num_train_samples}")
    print(f"Number of validation samples: {num_val_samples}")

    num_train_steps = (num_train_samples + batch_size - 1) // batch_size
    num_val_steps = (num_val_samples + batch_size - 1) // batch_size
    print(f"Steps per epoch (train): {num_train_steps}, Steps per epoch (val): {num_val_steps}")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=num_train_steps,
        validation_steps=num_val_steps,
        verbose=1,  # Print detailed log of training
        callbacks=[plot_metrics]
    )

    # Save metrics to CSV after training
    plot_metrics.save_to_csv(f'tf_loss/added_layer/{label_column}_{learning_rate}_metrics.csv')

    # Evaluate the model
    val_loss, val_mse, val_pearson = model.evaluate(val_dataset, steps=num_val_steps)
    print(f"Validation Loss: {val_loss}, Validation MSE: {val_mse}, Validation Pearson Correlation: {val_pearson}")

    # Extracting loss history
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    print(f"Training loss history: {train_loss}")
    print(f"Validation loss history: {val_loss}")

    # Debugging: Print lengths of loss arrays
    print(f"Length of train_loss: {len(train_loss)}, Length of val_loss: {len(val_loss)}")

    # Ensure lengths are the same before creating the text file
    if len(train_loss) != len(val_loss):
        raise ValueError(f"Length mismatch: train_loss ({len(train_loss)}) vs val_loss ({len(val_loss)})")

    # # Save loss data to a text file
    # loss_file_path = f'/camp/home/weie/lab_space_weie/training/transfer_learning/tf_loss/{label_column}_{learning_rate:.6f}_loss_history.txt'
    # with open(loss_file_path, 'w') as file:
    #     file.write("Epoch\tTraining Loss\tValidation Loss\n")
    #     for epoch in range(1, len(train_loss) + 1):
    #         file.write(f"{epoch}\t{train_loss[epoch-1]:.6f}\t{val_loss[epoch-1]:.6f}\n")
    # # Save loss data to a text file
    # loss_file_path = f'/camp/home/weie/lab_space_weie/training/transfer_learning/tf_loss/{label_column}_{learning_rate:.6f}_loss_history.txt'
    # with open(loss_file_path, 'w') as file:
    #     file.write("Epoch\tTraining Loss\tValidation Loss\n")
    #     for epoch in range(1, len(train_loss) + 1):
    #         file.write(f"{epoch}\t{train_loss[epoch-1]:.6f}\t{val_loss[epoch-1]:.6f}\n")

def fasta_generator(file_path, batch_size, max_length):
    def one_hot_encode_sequence(sequence, max_length):
        mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1]}
        encoded = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
        padding = [[0, 0, 0, 0]] * (max_length - len(encoded))
        return np.array(encoded + padding, dtype=np.float32)
    
    batch = []
    for record in SeqIO.parse(file_path, "fasta"):
        encoded_seq = one_hot_encode_sequence(str(record.seq), max_length)
        batch.append(encoded_seq)
        if len(batch) == batch_size:
            yield np.array(batch, dtype=np.float32)
            batch = []
    if batch:  # Yield remaining sequences if they don't fill the last batch
        while len(batch) < batch_size:
            batch.append(np.zeros((max_length, 4), dtype=np.float32))
        yield np.array(batch, dtype=np.float32)

def labels_generator(file_path, batch_size, label_column):
    labels = pd.read_csv(file_path, usecols=[label_column])[label_column].values
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i + batch_size]
        if len(batch_labels) < batch_size:  # Pad the labels to the batch size
            batch_labels = np.concatenate([batch_labels, np.zeros(batch_size - len(batch_labels))])
        yield batch_labels.reshape(-1, 1)

def load_labels_as_dataset(file_path, batch_size, label_column, shuffle=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: labels_generator(file_path, batch_size, label_column),
        output_types=tf.float32,
        output_shapes=(batch_size, 1)
)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)  # Shuffle with buffer size 10 times batch size
    return dataset.repeat()


def load_fasta_as_dataset(file_path, max_length, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: fasta_generator(file_path, batch_size, max_length),
        output_types=tf.float32,
        output_shapes=(batch_size, max_length, 4)
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)  # Shuffle with buffer size 10 times batch size
    return dataset.repeat().prefetch(buffer_size=1)

def load_combined_dataset(fasta_file_path, labels_file_path, max_length, batch_size, label_column, shuffle=False):
    sequence_dataset = load_fasta_as_dataset(fasta_file_path, max_length, batch_size, shuffle)
    labels_dataset = load_labels_as_dataset(labels_file_path, batch_size, label_column, shuffle)
    return tf.data.Dataset.zip((sequence_dataset, labels_dataset))

def pearson_correlation(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    covariance = tf.reduce_mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    y_true_var = tf.reduce_mean(tf.square(y_true - y_true_mean))
    y_pred_var = tf.reduce_mean(tf.square(y_pred - y_pred_mean))
    correlation = covariance / tf.sqrt(y_true_var * y_pred_var)
    return correlation

def count_samples(file_path):
    return sum(1 for line in open(file_path) if line.startswith(">"))

if __name__ == "__main__":
    main()