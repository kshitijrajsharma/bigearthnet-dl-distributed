"""Basic TensorFlow training with BigEarthNet Parquet data using Petastorm."""

import argparse
import tensorflow as tf
from petastorm import make_reader
from petastorm.tf_utils import make_petastorm_dataset

def train_model(data_path, epochs=10, batch_size=32, lr=0.001):
    """Train model using Petastorm reader."""
    
    # Create Petastorm reader for training data
    train_url = f"file://{data_path}" if not data_path.startswith('file://') and not data_path.startswith('s3://') else data_path
    
    print(f"Loading data from {train_url}")
    
    with make_reader(train_url, num_epochs=epochs) as reader:
        # Create TensorFlow dataset from Petastorm reader
        dataset = make_petastorm_dataset(reader)
        dataset = dataset.batch(batch_size)
        
        # Build simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(120, 120, 14)),  # 2 S1 bands + 12 S2 bands
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(43, activation='sigmoid'),  # Multi-label classification
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        # Prepare data generator
        def data_generator():
            for sample in reader:
                X = tf.concat([sample.s1_data, sample.s2_data], axis=-1)
                y = sample.reference
                yield X, y
        
        # Train
        history = model.fit(
            dataset,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
            ]
        )
        
        print("\nTraining complete!")
        print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return model, history

def main():
    parser = argparse.ArgumentParser(description='Train BigEarthNet model with TensorFlow')
    parser.add_argument('--data', required=True, help='Path to Parquet data directory or S3 path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save', type=str, help='Path to save model')
    
    args = parser.parse_args()
    
    model, history = train_model(args.data, args.epochs, args.batch, args.lr)
    
    if args.save:
        model.save(args.save)
        print(f"Model saved to {args.save}")

if __name__ == "__main__":
    main()
