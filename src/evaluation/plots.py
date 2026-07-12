"""Plotting functions for evaluations."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Set publication style styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plots and saves confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                annot_kws={"size": 12})
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    """Plots ROC Curve for a given model."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_pr_curve(y_true, y_prob, model_name="Model", save_path=None):
    """Plots Precision-Recall Curve for a given model."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="teal", lw=2, label=f"{model_name} (AUC = {pr_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left", frameon=True)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_fairness_accuracy_tradeoff(df_results, x_col="accuracy", y_col="demographic_parity_difference", save_path=None):
    """Plots Fairness vs. Accuracy tradeoff scatter plot."""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df_results,
        x=x_col,
        y=y_col,
        hue="model_name",
        style="mitigation_status" if "mitigation_status" in df_results.columns else None,
        s=120,
        palette="viridis",
        alpha=0.9
    )
    
    plt.xlabel("Model Accuracy (Higher is better)", fontsize=11, fontweight="bold")
    plt.ylabel("Demographic Parity Difference (Lower is better)", fontsize=11, fontweight="bold")
    plt.title("Fairness vs. Accuracy Tradeoff", fontsize=13, fontweight="bold", pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

