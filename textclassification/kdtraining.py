from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from dataloader import NewDataset
from kdtrainer import KDTrainer

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = f1_score(labels, preds, average = 'micro')
  return {
      'f1': acc,
  }




if __name__ == '__main__':
    max_length = 120
    train_path = './data/train.csv'
    dev_path = './data/dev.csv'
    test_path = './data/test.csv'
    label_set = ['severity','cause','treatment','method_diagnosis']
    student_model_name_or_path = ''
    teacher_model_name_or_path = ''


    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name_or_path,num_labels = len(label_set))
    student_model = AutoModelForSequenceClassification.from_pretrained(student_model_name_or_path,num_labels = len(label_set))
    
    train_dataset = NewDataset(max_length,tokenizer,train_path,label_set)
    valid_dataset = NewDataset(max_length,tokenizer,dev_path,label_set)
    test_dataset = NewDataset(max_length,tokenizer,test_path,label_set)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=5,              # total number of training epochs
        per_device_train_batch_size=256,  # batch size per device during training
        per_device_eval_batch_size=256,   # batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        metric_for_best_model='f1',
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=10,
        save_strategy="epoch",# log & save weights each logging_steps
        evaluation_strategy="epoch",     # evaluate each `logging_steps`
    )

    trainer = KDTrainer(
    teacher = teacher_model,
    selection_strategy = 'margin',
    selection_ratio = 0.6,
    model=student_model,                      # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )
    trainer.train()
    print(trainer.evaluate(eval_dataset=test_dataset))


