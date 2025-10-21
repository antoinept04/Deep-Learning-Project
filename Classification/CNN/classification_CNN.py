import torch



import wandb





import gc
import os
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler,Subset
import matplotlib.pyplot as plt
import cv2
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.v2 as transforms
from torch.nn import functional as F
from tqdm import tqdm
from torchsummary import summary
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger , CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
import pandas as pd
from torchmetrics.classification import MulticlassConfusionMatrix
from optuna.pruners import MedianPruner
import joblib

class Model(pl.LightningModule):
    def __init__(self, optimizer, num_classes=4, learning_rate=3e-4):
        super().__init__()
        
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
       
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.num_classes = num_classes
        self.optimizer = optimizer

        
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)  
        acc = self.val_acc(predicted, labels)
        self.log_dict({'train_loss':loss,"train_acc":acc}, on_step=True,prog_bar=True,logger=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.confusion_matrix.reset()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = self.val_acc(y_hat, y)
        self.log_dict({'val_loss':val_loss,"val_acc":val_acc}, on_step=False, on_epoch=True)
        self.confusion_matrix.update(y_hat.argmax(dim=1), y)
    def on_validation_epoch_end(self):
        self.val_acc.reset()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc = self.test_acc(y_hat, y)
        self.log_dict({'test_loss':test_loss,"test_acc":test_acc}, on_step=False, on_epoch=True)
        self.confusion_matrix.update(y_hat.argmax(dim=1), y)
        return test_loss

    def on_test_end(self):
        cm = self.confusion_matrix.compute().cpu().numpy()

        
        print("\n=== Confusion Matrix (test) ===")
        print(cm)

       
        self.confusion_matrix.reset()
        self.test_acc.reset()
        

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
    
def create_data_loaders(dataset_path, batch_size, train_split, val_split, img_size, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

   
    full_dataset = datasets.ImageFolder(dataset_path)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)


    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]


    train_dataset = Subset(datasets.ImageFolder(dataset_path, transform=train_transform), train_indices)
    val_dataset   = Subset(datasets.ImageFolder(dataset_path, transform=eval_transform ), val_indices)
    test_dataset  = Subset(datasets.ImageFolder(dataset_path, transform=eval_transform ), test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers= True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers= True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers= True
    )

    return train_loader, val_loader, test_loader




def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    epochs = trial.suggest_int("epochs", 5, 50)

    optimizer_class = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop
    }[optimizer_name]

    train_loader, val_loader, _ = create_data_loaders(dataset_path, batch_size, train_split, val_split, img_size)

    model = Model(optimizer=optimizer_class, num_classes=num_classes, learning_rate=learning_rate)


    csv_logger = CSVLogger(LOG_DIR, name=f"trial_{trial.number}")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    val_result = trainer.validate(model, val_loader)

    # Sauvegarde du modèle si c'est le meilleur
    joblib.dump(trial, f"{CHECKPOINT_DIR}/trial_{trial.number}.pkl")

    del model
    torch.cuda.empty_cache()
    gc.collect()


    return val_result[0]["val_loss"]


def predict_folder(model, folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print("Working dir:", os.getcwd())


    predictions = []
    filenames = []

    for img_path in os.listdir(folder_path):
        with torch.no_grad():
            # Load and preprocess image
            image = Image.open(folder_path+'/'+img_path).convert('RGB')
            image = test_transform(image).unsqueeze(0).to(device)

            # Get prediction
            output = model(image)
            _, predicted = output.max(1)

            predictions.append(predicted.item())
            filenames.append(img_path)

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'ID': filenames,
        'Label': predictions
    })

    # Save to CSV
    submission_df.to_csv('submissionbig.csv', index=False)
    return submission_df

    

if __name__ == "__main__":
    
    print("GPU disponible :", torch.cuda.is_available())
    print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU détecté")

    
    torch.set_float32_matmul_precision('medium')

    dataset_path = "C:/Intelligence artificielle/dataset/railway-construction-big"  #@param ["dataset/railway-construction-50/","dataset/railway-construction-100/","dataset/railway-construction-big/"] {type:"string"}
    batch_size = 32 #@param [8,16,32,64,128,256] {type:"raw"}
    train_split = 0.8 #@param {type:"slider", min:0.5, max:0.9, step:0.05}
    val_split = 0.1 #@param {type:"slider", min:0.1, max:0.5, step:0.05}
    epochs = 2 #@param [1,5, 10,20,50,100,200] {type:"raw"}
    learning_rate = 0.02  #@param [0.1, 0.01,0.02,0.05,0.001,0.002,0.005] {type:"raw"}
    img_size = 224
    num_classes = 4
    LOG_DIR = "logs/"

    
    wandb.login(key="1b255ea0fe77d23a9eba69f6fdcc5987fe897e35")
    # === PARAMÈTRES GLOBAUX ===
    wandb.init(project="Project_CNN")
    N_TRIALS = 10
    STUDY_NAME = "cnn_study2"
    STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # === CALLBACKS ===
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    # === ÉTUDE OPTUNA AVEC SAUVEGARDE + PRUNER ===
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5),
    )

    study.optimize(objective, n_trials=N_TRIALS)

    # === AFFICHAGE DES MEILLEURS PARAMÈTRES ===
    best_params = study.best_params
    print("Meilleurs hyperparamètres :", best_params)

    # === SAUVEGARDE DU MEILLEUR TRIAL ===
    joblib.dump(study.best_trial, f"{CHECKPOINT_DIR}/best_trial.pkl")

    # Chargement des données avec les meilleurs paramètres
    best_train_loader, best_val_loader, best_test_loader = create_data_loaders(
        dataset_path,
        best_params['batch_size'],
        train_split,
        val_split,
        img_size
    )

    # Logger Weights & Biases
    wandb_logger = WandbLogger(project="Project_CNN", name="BestCNN", log_model=True)

    # Création du modèle avec meilleurs hyperparamètres
    model = Model(
        optimizer={"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMSprop": torch.optim.RMSprop}[best_params['optimizer']],
        num_classes=num_classes,
        learning_rate=best_params['learning_rate'],
    )

    trainer = pl.Trainer(
        max_epochs=best_params['epochs'],
        accelerator="auto",
        logger=[wandb_logger, CSVLogger(LOG_DIR, name="cnn_best")],
        callbacks=[checkpoint_callback]
    )

    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_final.ckpt")

    if os.path.exists(final_checkpoint_path):
        print("✅ Modèle final déjà entraîné, chargement du checkpoint.")
        model = Model.load_from_checkpoint(final_checkpoint_path)
    else:
        print("🚀 Entraînement du meilleur modèle en cours...")
        trainer.fit(model, best_train_loader, best_val_loader)
        trainer.save_checkpoint(final_checkpoint_path)


    trainer.test(model, best_test_loader)

    wandb.finish()

    best_trial = joblib.load(f"{CHECKPOINT_DIR}/best_trial.pkl")
    print(best_trial.params)

    jit_model = model.to_torchscript()
    torch.jit.save(jit_model, 'model_jitb.pth')

    test_folder_path  = "C:/Intelligence artificielle/Test_path/test_no_classes (1)/test"

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    submission = predict_folder(jit_model, test_folder_path)

    from pytorch_bench import benchmark
    example_input = torch.randn(1, 3, 224, 224)
    results_pruned = benchmark(model, example_input, gpu_only=True)