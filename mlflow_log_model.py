"""mlflow_log_model.py

Trains a Keras model on hand landmark data and logs it to MLflow with accuracy & confusion matrix.

Usage (PowerShell):
    python mlflow_log_model.py --experiment hand_gesture_experiment
    python mlflow_log_model.py --epochs 50 --batch-size 32 --experiment my_experiment

This will:
 - Load training, validation, and test data from data/landmarks_*.npz
 - Train a new Keras model from scratch
 - Evaluate on test data and compute accuracy and confusion matrix
 - Log all metrics, artifacts, and the trained model to MLflow
"""

import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import NumPy
try:
    import numpy as np
except Exception as e:
    print("NumPy is required. Install with: pip install numpy")
    raise

# Import MLflow
try:
    import mlflow
except Exception as e:
    print("MLflow is not installed. Install with: pip install mlflow")
    raise

# Import TensorFlow (REQUIRED for model loading)
try:
    from keras.models import load_model
    import tensorflow as tf
    HAS_TF = True
except Exception as e:
    print(f"TensorFlow is required to load the .h5 model.")
    print(f"Install with: pip install tensorflow")
    sys.exit(1)


# Import scikit-learn for metrics
try:
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except Exception as e:
    print(f"Warning: scikit-learn not available: {e}")
    HAS_SKLEARN = False

# Import Keras layers for model building
try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from keras.utils import to_categorical
    HAS_KERAS = True
except Exception as e:
    print(f"Warning: Could not import Keras components: {e}")
    HAS_KERAS = False



def load_all_data(data_dir):
    """Load training, validation, and test data from landmarks_*.npz files.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test) or None if any file is missing.
    """
    try:
        train_path = os.path.join(data_dir, 'landmarks_train.npz')
        val_path = os.path.join(data_dir, 'landmarks_val.npz')
        test_path = os.path.join(data_dir, 'landmarks_test.npz')
        
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            print("ERROR: Missing data files!")
            print(f"  Train: {train_path} - {'OK' if os.path.exists(train_path) else 'MISSING'}")
            print(f"  Val:   {val_path} - {'OK' if os.path.exists(val_path) else 'MISSING'}")
            print(f"  Test:  {test_path} - {'OK' if os.path.exists(test_path) else 'MISSING'}")
            return None
        
        print("Loading training data...")
        train_data = np.load(train_path)
        X_train, y_train = train_data['X'], train_data['y']
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        
        print("Loading validation data...")
        val_data = np.load(val_path)
        X_val, y_val = val_data['X'], val_data['y']
        print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
        
        print("Loading test data...")
        test_data = np.load(test_path)
        X_test, y_test = test_data['X'], test_data['y']
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    except Exception as e:
        print(f"ERROR: Could not load data: {e}")
        return None


def encode_labels(y_train, y_val, y_test):
    """Encode string labels to numeric indices using LabelEncoder."""
    try:
        le = LabelEncoder()
        # Fit on all labels to ensure same encoding across sets
        all_labels = np.concatenate([y_train, y_val, y_test])
        le.fit(all_labels)
        
        y_train_encoded = le.transform(y_train)
        y_val_encoded = le.transform(y_val)
        y_test_encoded = le.transform(y_test)
        
        print(f"Encoded {len(le.classes_)} classes: {list(le.classes_)}")
        
        return y_train_encoded, y_val_encoded, y_test_encoded, le
    except Exception as e:
        print(f"ERROR: Could not encode labels: {e}")
        return None, None, None, None


def build_model(num_classes, input_dim=63):
    """Build a Keras Sequential model for hand gesture recognition."""
    try:
        model = Sequential([
            Dense(256, input_shape=(input_dim,)),
            BatchNormalization(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model built with {num_classes} output classes")
        return model
    except Exception as e:
        print(f"ERROR: Could not build model: {e}")
        return None


def train_model(model, X_train, y_train_cat, X_val, y_val_cat, epochs=50, batch_size=32):
    """Train the model and return training history."""
    try:
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            ModelCheckpoint('best_model.h5', save_best_only=True, verbose=0)
        ]
        
        print(f"\nTraining model for {epochs} epochs with batch_size={batch_size}...")
        history = model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        print("[TRAINING] Model training completed!")
        return history
    except Exception as e:
        print(f"ERROR: Could not train model: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_labels_to_numeric(y_test):
    """Convert string labels to numeric indices."""
    if y_test.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
        unique_labels = sorted(np.unique(y_test))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_numeric = np.array([label_to_idx[label] for label in y_test])
        return y_numeric, unique_labels
    else:
        return y_test.astype(int), None


def compute_metrics(model, X_test, y_test_numeric, le=None):
    """Compute accuracy, confusion matrix, and classification report."""
    if model is None or X_test is None or y_test_numeric is None:
        return None
    
    try:
        print("\nMaking predictions on test data...")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Compute accuracy
        accuracy = accuracy_score(y_test_numeric, y_pred)
        print(f"[TEST] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test_numeric, y_pred)
        print(f"[TEST] Confusion matrix: {cm.shape}")
        
        # Get classification report
        target_names = le.classes_.tolist() if le is not None else None
        report = classification_report(y_test_numeric, y_pred, 
                                     target_names=target_names,
                                     output_dict=True)
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'num_test_samples': len(X_test),
            'num_classes': cm.shape[0],
            'label_names': target_names
        }
    except Exception as e:
        print(f"ERROR: Could not compute metrics: {e}")
        import traceback
        traceback.print_exc()
        return None



def main(epochs, batch_size, experiment_name):
    
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)
    
    if not HAS_KERAS:
        print("ERROR: Keras components not available. Install with: pip install keras")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("[TRAINING PIPELINE] Hand Gesture Recognition Model")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*70}\n")
    
    project_dir = os.path.dirname(__file__)
    data_dir = os.path.join(project_dir, 'data')
    
    # Load all data
    print("STEP 1: Loading data")
    print("-" * 70)
    data = load_all_data(data_dir)
    if data is None:
        sys.exit(1)
    
    X_train, y_train, X_val, y_val, X_test, y_test = data
    
    # Encode labels
    print("\nSTEP 2: Encoding labels")
    print("-" * 70)
    y_train_enc, y_val_enc, y_test_enc, le = encode_labels(y_train, y_val, y_test)
    if le is None:
        sys.exit(1)
    
    num_classes = len(le.classes_)
    
    # Convert to categorical (one-hot)
    print("\nConverting to one-hot encoding...")
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_val_cat = to_categorical(y_val_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)
    print(f"  Train shape: {y_train_cat.shape}")
    print(f"  Val shape:   {y_val_cat.shape}")
    print(f"  Test shape:  {y_test_cat.shape}")
    
    # Build model
    print("\nSTEP 3: Building model")
    print("-" * 70)
    model = build_model(num_classes, input_dim=X_train.shape[1])
    if model is None:
        sys.exit(1)
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print("\nSTEP 4: Training model")
    print("-" * 70)
    history = train_model(model, X_train, y_train_cat, X_val, y_val_cat, 
                         epochs=epochs, batch_size=batch_size)
    if history is None:
        sys.exit(1)
    
    # Evaluate on test set
    print("\nSTEP 5: Evaluating on test set")
    print("-" * 70)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"[TEST] Loss: {test_loss:.4f}")
    print(f"[TEST] Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Compute detailed metrics
    print("\nSTEP 6: Computing detailed metrics")
    print("-" * 70)
    metrics = compute_metrics(model, X_test, y_test_enc, le)
    
    if metrics is None:
        print("WARNING: Could not compute detailed metrics")
        metrics = {}
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run and log
    print("\nSTEP 7: Logging to MLflow")
    print("-" * 70)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        
        # Log hyperparameters
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_param('num_training_samples', len(X_train))
        mlflow.log_param('num_validation_samples', len(X_val))
        mlflow.log_param('num_test_samples', len(X_test))
        mlflow.log_param('input_dim', X_train.shape[1])
        
        # Log environment info
        mlflow.log_param('tensorflow_version', tf.__version__)
        mlflow.log_param('mlflow_version', mlflow.__version__)
        mlflow.log_param('numpy_version', np.__version__)
        
        # Log metrics
        mlflow.log_metric('test_loss', float(test_loss))
        mlflow.log_metric('test_accuracy', float(test_acc))
        
        if metrics:
            mlflow.log_metric('accuracy_from_predictions', metrics['accuracy'])
        
        # Log training history metrics
        if history:
            final_train_acc = float(history.history['accuracy'][-1])
            final_val_acc = float(history.history['val_accuracy'][-1])
            final_train_loss = float(history.history['loss'][-1])
            final_val_loss = float(history.history['val_loss'][-1])
            
            mlflow.log_metric('final_train_accuracy', final_train_acc)
            mlflow.log_metric('final_val_accuracy', final_val_acc)
            mlflow.log_metric('final_train_loss', final_train_loss)
            mlflow.log_metric('final_val_loss', final_val_loss)
            
            print(f"\nLogging metrics:")
            print(f"  Train Accuracy: {final_train_acc:.4f}")
            print(f"  Val Accuracy:   {final_val_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
        
        # Log confusion matrix
        if metrics and 'confusion_matrix' in metrics:
            cm_file = 'confusion_matrix.json'
            with open(cm_file, 'w') as f:
                json.dump({
                    'confusion_matrix': metrics['confusion_matrix'],
                    'accuracy': metrics['accuracy'],
                    'num_classes': metrics['num_classes'],
                    'class_names': metrics.get('label_names')
                }, f, indent=2)
            mlflow.log_artifact(cm_file)
            print(f"  [OK] Logged: confusion_matrix.json")
            
            # Log classification report
            report_file = 'classification_report.json'
            with open(report_file, 'w') as f:
                json.dump(metrics['classification_report'], f, indent=2)
            mlflow.log_artifact(report_file)
            print(f"  [OK] Logged: classification_report.json")
        
        # Save and log the trained model
        model_path = 'best_model.h5'
        model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path='model')
        print(f"  [OK] Logged: model/best_model.h5")
        
        # Log label encoder
        import pickle
        encoder_file = 'label_encoder.pkl'
        with open(encoder_file, 'wb') as f:
            pickle.dump(le, f)
        mlflow.log_artifact(encoder_file, artifact_path='model')
        print(f"  [OK] Logged: model/label_encoder.pkl")
        
        print(f"\n{'='*70}")
        print("[SUCCESS] Training pipeline completed!")
        print(f"{'='*70}")
        print(f"Run ID: {run_id}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        if metrics:
            print(f"Classes: {metrics['num_classes']}")
        print(f"\nView results with: mlflow ui --port 5000")
        print(f"{'='*70}\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hand gesture model and log to MLflow')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--experiment', type=str, default='hand_gesture_training', help='MLflow experiment name (default: hand_gesture_training)')
    args = parser.parse_args()
    
    main(args.epochs, args.batch_size, args.experiment)


