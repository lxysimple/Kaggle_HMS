import tensorflow as tf
print(tf.__version__)

##################################################################################

# Set to True for inference only, False for training
# ONLY_INFERENCE = True
ONLY_INFERENCE = False

# Configuration for model training
FOLDS = 5
EPOCHS = 4
BATCH = 32
NAME = 'None'

SPEC_SIZE  = (512, 512, 3)
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
TARGETS = CLASSES

##################################################################################

import gc
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras import backend as K
from tqdm import tqdm 
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import spectrogram as spectrogram_np

import efficientnet.tfkeras as efn

sys.path.append(f'/home/xyli/kaggle/')
from kaggle_kl_div import score

##################################################################################

sys.path += ["/opt/conda/envs/rapids/lib/python3.7/site-packages"]
sys.path += ["/opt/conda/envs/rapids/lib/python3.7"]
sys.path += ["/opt/conda/envs/rapids/lib"]

import cupy as cp
import cusignal

##################################################################################

# Set the visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

# Set the strategy for using GPUs
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')

# Configure memory growth
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Enable or disable mixed precision
MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')
else:
    print('Using full precision')


##################################################################################
    
# Function to set random seed for reproducibility
def set_random_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        os.environ.pop('TF_DETERMINISTIC_OPS', None)

# Set a deterministic behavior
set_random_seed(deterministic=True)

##################################################################################

def create_train_data():
    # Read the dataset
    df = pd.read_csv('/home/xyli/kaggle/train.csv')
    
    # Create a new identifier combining multiple columns
    id_cols = ['eeg_id', 'spectrogram_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df['new_id'] = df[id_cols].astype(str).agg('_'.join, axis=1)
    
    # Calculate the sum of votes for each class
    df['sum_votes'] = df[CLASSES].sum(axis=1)
    
    # Group the data by the new identifier and aggregate various features
    agg_functions = {
        'eeg_id': 'first',
        'eeg_label_offset_seconds': ['min', 'max'],
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'spectrogram_id': 'first',
        'patient_id': 'first',
        'expert_consensus': 'first',
        **{col: 'sum' for col in CLASSES},
        'sum_votes': 'mean',
    }
    grouped_df = df.groupby('new_id').agg(agg_functions).reset_index()

    # Flatten the MultiIndex columns and adjust column names
    grouped_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped_df.columns]
    grouped_df.columns = grouped_df.columns.str.replace('_first', '').str.replace('_sum', '').str.replace('_mean', '')
    
    # Normalize the class columns
    y_data = grouped_df[CLASSES].values
    y_data_normalized = y_data / y_data.sum(axis=1, keepdims=True)
    grouped_df[CLASSES] = y_data_normalized

    # Split the dataset into high and low quality based on the sum of votes
    high_quality_df = grouped_df[grouped_df['sum_votes'] >= 10].reset_index(drop=True)
    low_quality_df = grouped_df[(grouped_df['sum_votes'] < 10) & (grouped_df['sum_votes'] >= 0)].reset_index(drop=True)

    return high_quality_df, low_quality_df

##################################################################################
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size=32, shuffle=False, mode='train'):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        # Initialization
        X = np.zeros((len(indexes), *SPEC_SIZE), dtype='float32')
        y = np.zeros((len(indexes), len(CLASSES)), dtype='float32')

        # Generate data
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            eeg_id = row['eeg_id']
            spec_offset = int(row['spectrogram_label_offset_seconds_min'])
            eeg_offset = int(row['eeg_label_offset_seconds_min'])
            file_path = f'/kaggle/input/3-diff-time-specs-hms/images/{eeg_id}_{spec_offset}_{eeg_offset}.npz'
            data = np.load(file_path)
            eeg_data = data['final_image']
            eeg_data_expanded = np.repeat(eeg_data[:, :, np.newaxis], 3, axis=2)

            X[j] = eeg_data_expanded
            if self.mode != 'test':
                y[j] = row[CLASSES]

        return X, y

##################################################################################
def lrfn(epoch):
    lr_schedule = [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5]
    return lr_schedule[epoch]

# Define the learning rate scheduler callback
LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

def build_EfficientNetB0(input_shape=(512, 512, 3), num_classes=6):
    inp = tf.keras.Input(shape=input_shape)

    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    base_model.load_weights(f'/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')

    # OUTPUT
    x = base_model(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes,activation='softmax', dtype='float32')(x)

    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt)

    return model
##################################################################################
def cross_validate_model(train_data, train_data_2, folds, random_seed, targets, nome_modelo):
    inicio = time.time()
    path_model = f'MLP_Model{nome_modelo}'
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    all_oof = []
    all_oof2 = []
    all_true = []
    models = []
    score_list = []
    
    # Separating the data to iterate over both dataframes simultaneously
    gkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    splits1 = list(gkf.split(train_data, train_data[["expert_consensus"]], train_data["patient_id"]))
    splits2 = list(gkf.split(train_data_2, train_data_2[["expert_consensus"]], train_data_2["patient_id"]))

    # Iterate over folds in parallel
    for i, ((train_index, valid_index), (train_index2, valid_index2)) in enumerate(zip(splits1, splits2)):
        
        # Copy the dataframes to avoid leaks
        train_data_ = train_data.copy()
        train_data_2_ = train_data_2.copy()
        set_random_seed(random_seed, deterministic=True)
        
        # Start folding
        print('#' * 25)
        print(f'### Fold {i + 1}')
        print(f'### train size 1 {len(train_index)}, valid size {len(valid_index)}')
        print(f'### train size 2 {len(train_index2)}, valid size {len(valid_index2)}')
        print('#' * 25)

        ### --------------------------- Performs model 1 training -------------- --------------------------- ###
        K.clear_session()
        train_gen = DataGenerator(train_data_.iloc[train_index], shuffle=True, batch_size=BATCH)
        valid_gen = DataGenerator(train_data_.iloc[valid_index], shuffle=False, batch_size=(BATCH*2), mode='valid')
        model = build_EfficientNetB0(input_shape=(512, 512, 3), num_classes=6)
        history = model.fit(train_gen, verbose=2, validation_data=valid_gen, epochs=EPOCHS, callbacks=[LR])

        # Model training result 1
        train_loss = history.history['loss'][-1]  
        valid_loss = history.history['val_loss'][-1]
        print(f'train_loss 1 {train_loss} valid_loss 1 {valid_loss}')
        score_list.append((train_loss, valid_loss))

        
        ### --------------------------- creation of pseudo labels ---------------- ------------------------- ###
        # pseudo labels for low quality data
        train_2_index_total_gen = DataGenerator(train_data_2_.iloc[train_index2], shuffle=False, batch_size=BATCH)
        pseudo_labels_2 = model.predict(train_2_index_total_gen, verbose=2)
        # Refinement of low quality labels
        train_data_2_.loc[train_index2, TARGETS] /= 2
        train_data_2_.loc[train_index2, TARGETS] += pseudo_labels_2 / 2

        # pseudo labels for high quality data (50% of data)
        train_data_3_ = train_data_
        train_3_index_total_gen = DataGenerator(train_data_3_.iloc[train_index], shuffle=False, batch_size=BATCH)
        pseudo_labels_3 = model.predict(train_3_index_total_gen, verbose=2)
        # Refinement of high quality labels
        train_data_3_.loc[train_index, TARGETS] /= 2
        train_data_3_.loc[train_index, TARGETS] += pseudo_labels_3 / 2

        ### --------------------------- Creation of the data generator for the refined labels model --------- -------------------------------- ###
        # Low quality data
        np.random.shuffle(train_index)
        np.random.shuffle(valid_index)
        sixty_percent_length = int(0.5 * len(train_data_3_))
        train_index_60 = train_index[:int(sixty_percent_length * len(train_index) / len(train_data_3_))]
        valid_index_60 = valid_index[:int(sixty_percent_length * len(valid_index) / len(train_data_3_))]
        train_gen_2 = DataGenerator(pd.concat([train_data_3_.iloc[train_index_60], train_data_2_.iloc[train_index2]]), shuffle=True, batch_size=BATCH)
        valid_gen_2 = DataGenerator(pd.concat([train_data_3_.iloc[valid_index_60], train_data_2_.iloc[valid_index2]]), shuffle=False, batch_size=BATCH*2, mode='valid')
        # Rebuild the high quality data generator with 50% of the labels refined
        train_gen = DataGenerator(train_data_.iloc[train_index], shuffle=True, batch_size=BATCH)
        valid_gen = DataGenerator(train_data_.iloc[valid_index], shuffle=False, batch_size=(BATCH*2), mode='valid')
        
        ### --------------------------- Model 2 training and finetunning -------------- --------------------------- ###
        K.clear_session()
        new_model = build_EfficientNetB0(input_shape=(512, 512, 3), num_classes=6)
        # Training with the refined low-quality data
        history = new_model.fit(train_gen_2, verbose=2, validation_data=valid_gen_2, epochs=EPOCHS, callbacks=[LR])
        # Finetuning with refined high-quality data
        history = new_model.fit(train_gen, verbose=2, validation_data=valid_gen, epochs=EPOCHS, callbacks=[LR])
        new_model.save_weights(f'{path_model}/MLP_fold{i}.weights.h5')
        models.append(new_model)

        # Model 2 training result
        train_loss = history.history['loss'][-1]  # Valor da perda do último epoch de treinamento
        valid_loss = history.history['val_loss'][-1]  # Valor da perda do último epoch de validação
        print(f'train_loss 2 {train_loss} valid_loss 2 {valid_loss}')
        score_list.append((train_loss, valid_loss))


        # MLP OOF
        oof = new_model.predict(valid_gen, verbose=2)
        all_oof.append(oof)
        all_true.append(train_data.iloc[valid_index][TARGETS].values)

        # TRAIN MEAN OOF
        y_train = train_data.iloc[train_index][targets].values
        y_valid = train_data.iloc[valid_index][targets].values
        oof = y_valid.copy()
        for j in range(6):
            oof[:,j] = y_train[:,j].mean()
        oof = oof / oof.sum(axis=1,keepdims=True)
        all_oof2.append(oof)

        del model, new_model, train_gen, valid_gen, train_2_index_total_gen, train_gen_2, valid_gen_2, oof, y_train, y_valid, train_index, valid_index
        K.clear_session()
        gc.collect()

        if i==folds-1: break

    all_oof = np.concatenate(all_oof)
    all_oof2 = np.concatenate(all_oof2)
    all_true = np.concatenate(all_true)

    oof = pd.DataFrame(all_oof.copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true['id'] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name='id')
    fim = time.time()
    tempo_execucao = fim - inicio
    print(f'{nome_modelo} CV Score with EEG Spectrograms ={cv} tempo: {tempo_execucao}')
    
    gc.collect()

    score_array = np.array(score_list)
    std_dev = np.std(score_array, axis=0)
    std_dev = std_dev.tolist()

    return cv, tempo_execucao, all_oof, all_oof2, all_true, models, score_list, std_dev, path_model

##################################################################################
if not ONLY_INFERENCE:
    high_quality_df, low_quality_df = create_train_data()
    result, tempo_execucao, all_oof, all_oof2, all_true, models, score_list, std_dev, path_model = cross_validate_model(high_quality_df, low_quality_df, FOLDS, 42, CLASSES, NAME)
    print(f'Result cv V1 final {result}{tempo_execucao} {score_list} {std_dev}')
    display(result)

##################################################################################
