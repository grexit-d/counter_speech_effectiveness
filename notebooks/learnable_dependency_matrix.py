from utils.helpers import set_seed, FocalLoss, MultiTaskDataset, LearnableDependencyMatrix, preprocess_data
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd


def evaluate_model(model, test_loader, device, multitask=None, output_file="dm_learnable_6e_traintest_twitter.txt"):
    """
    Evaluate the multi-task BERT model on the test set.

    Args:
        model: Trained multi-task BERT model.
        test_loader: DataLoader for the test set.
        device: The device to run the evaluation on ('cuda' or 'cpu').
        output_file: File path to save the evaluation results.
    """
    model.eval()
    all_preds = { "emotional": [], "audience": [], "clarity": [], "evidence": [], "rebuttal": [], "fairness": [] }
    all_labels = { "emotional": [], "audience": [], "clarity": [], "evidence": [], "rebuttal": [], "fairness": [] }

    # Evaluation loop
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Ground truth labels
            labels_emotional = batch['labels_emotional'].to(device)
            labels_audience = batch['labels_audience'].to(device)
            labels_clarity = batch['labels_clarity'].to(device)
            labels_evidence = batch['labels_evidence'].to(device)
            labels_rebuttal = batch['labels_rebuttal'].to(device)
            labels_fairness = batch['labels_fairness'].to(device)

            if args.multitask == "united":
                emotional_logits, audience_logits, clarity_logits, evidence_logits, rebuttal_logits, fairness_logits = model(input_ids, attention_mask)
                # Convert logits to predictions
                preds_emotional = (torch.sigmoid(emotional_logits) > 0.5).int().cpu().numpy()
                preds_audience = (torch.sigmoid(audience_logits) > 0.5).int().cpu().numpy()
                preds_clarity = torch.argmax(clarity_logits, dim=1).cpu().numpy()
                preds_evidence = torch.argmax(evidence_logits, dim=1).cpu().numpy()
                preds_rebuttal = torch.argmax(rebuttal_logits, dim=1).cpu().numpy()
                preds_fairness = torch.argmax(fairness_logits, dim=1).cpu().numpy()

                # Append predictions and true labels
                all_preds["emotional"].extend(preds_emotional)
                all_preds["audience"].extend(preds_audience)
                all_preds["clarity"].extend(preds_clarity)
                all_preds["evidence"].extend(preds_evidence)
                all_preds["rebuttal"].extend(preds_rebuttal)
                all_preds["fairness"].extend(preds_fairness)

                all_labels["emotional"].extend(labels_emotional.cpu().numpy())
                all_labels["audience"].extend(labels_audience.cpu().numpy())
                all_labels["clarity"].extend(labels_clarity.cpu().numpy())
                all_labels["evidence"].extend(labels_evidence.cpu().numpy())
                all_labels["rebuttal"].extend(labels_rebuttal.cpu().numpy())
                all_labels["fairness"].extend(labels_fairness.cpu().numpy())
                
            elif args.multitask == "separated-binary":
                emotional_logits, audience_logits = model(input_ids, attention_mask)
                preds_emotional = (torch.sigmoid(emotional_logits) > 0.5).int().cpu().numpy()
                preds_audience = (torch.sigmoid(audience_logits) > 0.5).int().cpu().numpy()

                all_preds["emotional"].extend(preds_emotional)
                all_preds["audience"].extend(preds_audience)
    
                all_labels["emotional"].extend(labels_emotional.cpu().numpy())
                all_labels["audience"].extend(labels_audience.cpu().numpy())
            elif args.multitask == "separated-multi":
                clarity_logits, evidence_logits, rebuttal_logits, fairness_logits = model(input_ids, attention_mask)

                preds_clarity = torch.argmax(clarity_logits, dim=1).cpu().numpy()
                preds_evidence = torch.argmax(evidence_logits, dim=1).cpu().numpy()
                preds_rebuttal = torch.argmax(rebuttal_logits, dim=1).cpu().numpy()
                preds_fairness = torch.argmax(fairness_logits, dim=1).cpu().numpy()

                all_preds["clarity"].extend(preds_clarity)
                all_preds["evidence"].extend(preds_evidence)
                all_preds["rebuttal"].extend(preds_rebuttal)
                all_preds["fairness"].extend(preds_fairness)

                all_labels["clarity"].extend(labels_clarity.cpu().numpy())
                all_labels["evidence"].extend(labels_evidence.cpu().numpy())
                all_labels["rebuttal"].extend(labels_rebuttal.cpu().numpy())
                all_labels["fairness"].extend(labels_fairness.cpu().numpy())
            else:
                # Model predictions
                #emotional_logits, audience_logits, clarity_logits, evidence_logits, rebuttal_logits, fairness_logits = model(input_ids, attention_mask)
                (emotional_logits, audience_logits, clarity_logits,
                evidence_logits, rebuttal_logits, fairness_logits), _ = model(input_ids, attention_mask)

                # Convert logits to predictions
                preds_emotional = (torch.sigmoid(emotional_logits) > 0.5).int().cpu().numpy()
                preds_audience = (torch.sigmoid(audience_logits) > 0.5).int().cpu().numpy()
                preds_clarity = torch.argmax(clarity_logits, dim=1).cpu().numpy()
                preds_evidence = torch.argmax(evidence_logits, dim=1).cpu().numpy()
                preds_rebuttal = torch.argmax(rebuttal_logits, dim=1).cpu().numpy()
                preds_fairness = torch.argmax(fairness_logits, dim=1).cpu().numpy()

                # Append predictions and true labels
                all_preds["emotional"].extend(preds_emotional)
                all_preds["audience"].extend(preds_audience)
                all_preds["clarity"].extend(preds_clarity)
                all_preds["evidence"].extend(preds_evidence)
                all_preds["rebuttal"].extend(preds_rebuttal)
                all_preds["fairness"].extend(preds_fairness)

                all_labels["emotional"].extend(labels_emotional.cpu().numpy())
                all_labels["audience"].extend(labels_audience.cpu().numpy())
                all_labels["clarity"].extend(labels_clarity.cpu().numpy())
                all_labels["evidence"].extend(labels_evidence.cpu().numpy())
                all_labels["rebuttal"].extend(labels_rebuttal.cpu().numpy())
                all_labels["fairness"].extend(labels_fairness.cpu().numpy())


    all_preds = {metrics: xs for metrics, xs in all_preds.items() if len(xs) > 0}
    all_labels = {metrics: xs for metrics, xs in all_labels.items() if len(xs) > 0}

    # Initialize results dictionary
    results = {}

    # Calculate metrics for each dimension
    for dimension in all_preds.keys():
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels[dimension], all_preds[dimension], average="weighted"
        )
        classification_rep = classification_report(
            all_labels[dimension], all_preds[dimension]
        )
        results[dimension] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": classification_rep,
        }

    # Save results to a file
    with open(output_file, "w") as f:
        for dimension, metrics in results.items():
            f.write(f"### Evaluation for {dimension.capitalize()} ###\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(metrics["classification_report"])
            f.write("\n\n")

    print(f"Evaluation results saved to {output_file}")


# Training Loop
def train_model(model, train_loader, optimizer, focal_loss, num_epochs=6, device="cpu", multitask=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_emotional = batch['labels_emotional'].to(device)
            labels_audience = batch['labels_audience'].to(device)
            labels_clarity = batch['labels_clarity'].to(device)
            labels_evidence = batch['labels_evidence'].to(device)
            labels_rebuttal = batch['labels_rebuttal'].to(device)
            labels_fairness = batch['labels_fairness'].to(device)
            
            
            if multitask == "united":
                emotional_logits, audience_logits, clarity_logits, evidence_logits, rebuttal_logits, fairness_logits = model(input_ids, attention_mask)

                emotional_loss = nn.BCEWithLogitsLoss()(emotional_logits, labels_emotional)
                audience_loss = focal_loss(audience_logits, labels_audience)
                clarity_loss = nn.CrossEntropyLoss()(clarity_logits, labels_clarity)
                evidence_loss = nn.CrossEntropyLoss()(evidence_logits, labels_evidence)
                rebuttal_loss = nn.CrossEntropyLoss()(rebuttal_logits, labels_rebuttal)
                fairness_loss = nn.CrossEntropyLoss()(fairness_logits, labels_fairness)
                loss = emotional_loss + audience_loss + clarity_loss + evidence_loss + rebuttal_loss + fairness_loss
                total_loss += loss.item()
                loss.backward()

            elif multitask == "separated-binary":
                emotional_logits, audience_logits = model(input_ids, attention_mask)

                emotional_loss = nn.BCEWithLogitsLoss()(emotional_logits, labels_emotional)
                audience_loss = focal_loss(audience_logits, labels_audience)

                loss = emotional_loss + audience_loss
                total_loss += loss.item()
                loss.backward()
            elif multitask == "separated-multi":
                clarity_logits, evidence_logits, rebuttal_logits, fairness_logits = model(input_ids, attention_mask)

                clarity_loss = nn.CrossEntropyLoss()(clarity_logits, labels_clarity)
                evidence_loss = nn.CrossEntropyLoss()(evidence_logits, labels_evidence)
                rebuttal_loss = nn.CrossEntropyLoss()(rebuttal_logits, labels_rebuttal)
                fairness_loss = nn.CrossEntropyLoss()(fairness_logits, labels_fairness)

                loss = clarity_loss + evidence_loss + rebuttal_loss + fairness_loss
                total_loss += loss.item()
                loss.backward()
            else:
                task_outputs, dependency_matrix = model(input_ids, attention_mask)
                emotional_logits, audience_logits, clarity_logits, evidence_logits, rebuttal_logits, fairness_logits = task_outputs

                emotional_loss = nn.BCEWithLogitsLoss()(emotional_logits, labels_emotional)
                audience_loss = focal_loss(audience_logits, labels_audience)
                clarity_loss = nn.CrossEntropyLoss()(clarity_logits, labels_clarity)
                evidence_loss = nn.CrossEntropyLoss()(evidence_logits, labels_evidence)
                rebuttal_loss = nn.CrossEntropyLoss()(rebuttal_logits, labels_rebuttal)
                fairness_loss = nn.CrossEntropyLoss()(fairness_logits, labels_fairness)

                task_losses = [emotional_loss, audience_loss, clarity_loss, evidence_loss, rebuttal_loss, fairness_loss]
                weighted_loss = sum(
                    dependency_matrix[i, j] * task_losses[i]
                    for i in range(len(task_losses))
                    for j in range(len(task_losses))
                )
                total_loss += weighted_loss.item()
                weighted_loss.backward()            

            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")


# Multi-Task BERT Model
class MultiTaskBERT(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_tasks=6):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.emotional_appeal_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.audience_adaptation_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.clarity_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.evidence_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.rebuttal_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.fairness_head = nn.Linear(self.bert.config.hidden_size, 3)

        # Learnable dependency matrix
        self.dependency_matrix = LearnableDependencyMatrix(num_tasks)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        # Task-specific outputs
        emotional_output = self.emotional_appeal_head(pooled_output)
        audience_output = self.audience_adaptation_head(pooled_output)
        clarity_output = self.clarity_head(pooled_output)
        evidence_output = self.evidence_head(pooled_output)
        rebuttal_output = self.rebuttal_head(pooled_output)
        fairness_output = self.fairness_head(pooled_output)

        return (
            emotional_output, audience_output, clarity_output, evidence_output, rebuttal_output, fairness_output
        ), self.dependency_matrix()


# Multi-Task BERT Model
class MultiTaskBERTUnited(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_tasks=6):
        super(MultiTaskBERTUnited, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.emotional_appeal_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.audience_adaptation_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.clarity_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.evidence_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.rebuttal_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.fairness_head = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        # Task-specific outputs
        emotional_output = self.emotional_appeal_head(pooled_output)
        audience_output = self.audience_adaptation_head(pooled_output)
        clarity_output = self.clarity_head(pooled_output)
        evidence_output = self.evidence_head(pooled_output)
        rebuttal_output = self.rebuttal_head(pooled_output)
        fairness_output = self.fairness_head(pooled_output)

        return emotional_output, audience_output, clarity_output, evidence_output, rebuttal_output, fairness_output


# Multi-Task BERT Model
class MultiTaskBERTBinary(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_tasks=6):
        super(MultiTaskBERTBinary, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.emotional_appeal_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.audience_adaptation_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        # Task-specific outputs
        emotional_output = self.emotional_appeal_head(pooled_output)
        audience_output = self.audience_adaptation_head(pooled_output)

        return emotional_output, audience_output


# Multi-Task BERT Model
class MultiTaskBERTMultiClass(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_tasks=6):
        super(MultiTaskBERTMultiClass, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)

        self.clarity_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.evidence_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.rebuttal_head = nn.Linear(self.bert.config.hidden_size, 3)
        self.fairness_head = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        # Task-specific outputs
        clarity_output = self.clarity_head(pooled_output)
        evidence_output = self.evidence_head(pooled_output)
        rebuttal_output = self.rebuttal_head(pooled_output)
        fairness_output = self.fairness_head(pooled_output)

        #return emotional_output, audience_output, clarity_output, evidence_output, rebuttal_output, fairness_output
        return clarity_output, evidence_output, rebuttal_output, fairness_output


def main(args):
    # Set seed
    set_seed(args.seed)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data1 = pd.read_csv(args.train_file)
    data2 = pd.read_csv(args.eval_file)

    # If combined, split 70-10-20
    if args.combine:
        combined_data = pd.concat([data1, data2], ignore_index=True)

        # Preprocess the combined dataset
        input_ids, attention_mask, labels_emotional, labels_audience, labels_clarity, labels_evidence, labels_rebuttal, labels_fairness = preprocess_data(combined_data, tokenizer, only_cs=args.only_cs)

        temp_ids, test_ids, temp_mask, test_mask, temp_emotional, test_emotional, temp_audience, test_audience, temp_clarity, test_clarity, temp_evidence, test_evidence, temp_rebuttal, test_rebuttal, temp_fairness, test_fairness = train_test_split(
            input_ids, attention_mask, labels_emotional, labels_audience, labels_clarity, labels_evidence, labels_rebuttal, labels_fairness, test_size=0.2, random_state=args.seed
        )

        train_ids, val_ids, train_mask, val_mask, train_emotional, val_emotional, train_audience, val_audience, train_clarity, val_clarity, train_evidence, val_evidence, train_rebuttal, val_rebuttal, train_fairness, val_fairness = train_test_split(
            temp_ids, temp_mask, temp_emotional, temp_audience, temp_clarity, temp_evidence, temp_rebuttal, temp_fairness, test_size=0.125, random_state=args.seed
        )
    # If cross-evaluation, split 80-20 the in dataset and test on everything from
    # the other one
    elif args.train_file == args.eval_file:
        input_ids, attention_mask, labels_emotional, labels_audience, labels_clarity, labels_evidence, labels_rebuttal, labels_fairness = preprocess_data(data1, tokenizer, only_cs=args.only_cs)

        train_ids, temp_ids, train_mask, temp_mask, train_emotional, temp_emotional, train_audience, temp_audience, train_clarity, temp_clarity, train_evidence, temp_evidence, train_rebuttal, temp_rebuttal, train_fairness, temp_fairness = train_test_split(
            input_ids, attention_mask, labels_emotional, labels_audience, labels_clarity, labels_evidence, labels_rebuttal, labels_fairness, test_size=0.2, random_state=args.seed
        )

        val_ids, test_ids, val_mask, test_mask, val_emotional, test_emotional, val_audience, test_audience, val_clarity, test_clarity, val_evidence, test_evidence, val_rebuttal, test_rebuttal, val_fairness, test_fairness = train_test_split(
            temp_ids, temp_mask, temp_emotional, temp_audience, temp_clarity, temp_evidence, temp_rebuttal, temp_fairness, test_size=0.125, random_state=args.seed
        )
    else:
        input_ids, attention_mask, labels_emotional, labels_audience, labels_clarity, labels_evidence, labels_rebuttal, labels_fairness = preprocess_data(data1, tokenizer, only_cs=args.only_cs)

        train_ids, val_ids, train_mask, val_mask, train_emotional, val_emotional, train_audience, val_audience, train_clarity, val_clarity, train_evidence, val_evidence, train_rebuttal, val_rebuttal, train_fairness, val_fairness = train_test_split(
            input_ids, attention_mask, labels_emotional, labels_audience, labels_clarity, labels_evidence, labels_rebuttal, labels_fairness, test_size=0.2, random_state=args.seed
        )

        test_ids, test_mask, test_emotional, test_audience, test_clarity, test_evidence, test_rebuttal, test_fairness = preprocess_data(data2, tokenizer, only_cs=args.only_cs)

    # Create DataLoaders
    train_dataset = MultiTaskDataset(train_ids, train_mask, train_emotional, train_audience, train_clarity, train_evidence, train_rebuttal, train_fairness)
    val_dataset = MultiTaskDataset(val_ids, val_mask, val_emotional, val_audience, val_clarity, val_evidence, val_rebuttal, val_fairness)
    test_dataset = MultiTaskDataset(test_ids, test_mask, test_emotional, test_audience, test_clarity, test_evidence, test_rebuttal, test_fairness)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, Optimizer, and Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.multitask == "united":
        model = MultiTaskBERTUnited(num_tasks=6).to(device)
    elif args.multitask == "separated-binary":
        model = MultiTaskBERTBinary(num_tasks=2).to(device)
    elif args.multitask == "separated-multi":
        model = MultiTaskBERTMultiClass(num_tasks=4).to(device)
    else:
        model = MultiTaskBERT(num_tasks=6).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    focal_loss = FocalLoss(alpha=1, gamma=2)

    # Train the model
    train_model(
        model,
        train_loader,
        optimizer,
        focal_loss,
        num_epochs=args.num_epochs,
        device=device,
        multitask=args.multitask,
    )

    if args.combine:
        output_file = f"multitask={args.multitask}__seed={args.seed}__{args.num_epochs}e__combined__only-cs={args.only_cs}.txt"
    else:
        output_file = f"multitask={args.multitask}__seed={args.seed}__{args.num_epochs}e__only-cs={args.only_cs}__train={args.train_file.strip('.csv').split('/')[-1]}__test={args.eval_file.strip('.csv').split('/')[-1]}.txt"

    evaluate_model(
        model,
        test_loader,
        device,
        multitask=args.multitask,
        output_file=f"./results/{output_file}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task BERT training")

    parser.add_argument('--train-file', type=str, required=True, help='Path to training dataset (CSV)')
    parser.add_argument('--eval-file', type=str, required=True, help='Path to evaluation dataset (CSV)')
    parser.add_argument('--combine', action='store_true', help='Whether to combine train and eval datasets')
    parser.add_argument(
        "--multitask",
        default=None,
        choices=[
            "united",
            "separated-binary",
            "separated-multi",
        ],
        help="Use multitask united only",
    )
    parser.add_argument('--only-cs', action='store_true', help='Use only CS')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=6, help='Num epochs')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')

    args = parser.parse_args()
    main(args)
