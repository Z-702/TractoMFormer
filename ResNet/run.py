import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import get_loaders
import logging

# ================== 配置 ==================
root_dir = "/data01/zixi/CNP_tractoembedding"
csv_path = "/data01/zixi/TractoFormer/TractoFormer-MVIT-main/CNP_150.csv"
module = "density"          # 可改成 density / trace1 /FA1
size = 320              # 图像大小
batch_size = 8
num_epochs = 50         # 每折训练的 epoch
lr = 1e-4
k_folds = 5             # 5 折交叉验证
weight_decay = 1e-4     # 正则化

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 日志 ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("train_5fold.log"), logging.StreamHandler()]
)

# ================== 存结果 ==================
all_results = []  # (fold, best_val_acc, best_epoch, model_path)

for val_fold in range(k_folds):
    logging.info(f"\n========== Fold {val_fold} ==========")

    # ---------- DataLoader ----------
    train_loader, val_loader = get_loaders(
        root_dir, csv_path,
        module=module, size=size,
        val_fold=val_fold, batch_size=batch_size
    )

    # ---------- 模型 ----------
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),              # dropout 防止过拟合
        nn.Linear(num_features, 2)
    )
    model = model.to(device)

    # ---------- 损失函数 & 优化器 ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------- 记录最佳 ----------
    best_val_acc, best_epoch = 0.0, -1
    best_model_path = f"best_fold{val_fold}.pth"

    # ---------- 训练 ----------
    for epoch in range(num_epochs):
        # --- train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- validate ---
        model.eval()
        val_correct, val_total, val_loss_total = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        # 更新最佳
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)  # 保存模型

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} "
            f"|| Best Epoch: {best_epoch}, Best Val Acc: {best_val_acc:.4f}"
        )

    logging.info(f"✅ Fold {val_fold} Finished | Best Val Acc = {best_val_acc:.4f} at Epoch {best_epoch}")
    all_results.append((val_fold, best_val_acc, best_epoch, best_model_path))

# ================== 选出最优 fold ==================
best_fold, best_acc, best_epoch, best_model_path = max(all_results, key=lambda x: x[1])
logging.info(f"\n🌟 Best Fold: {best_fold} | Val Acc = {best_acc:.4f} at Epoch {best_epoch}")
logging.info(f"🌟 Best Model Saved at: {best_model_path}")
