import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import TrainingArguments, Trainer

from datasets import load_dataset, load_metric, load_from_disk
import datasets
import numpy as np
import torch
import time
from models.CNN_test.layers import Linear

import numpy

def copy_lin_layer(old_layer, in_f, out_f):
    new_lin = Linear(in_features=in_f, out_features=out_f, bias=True)
    new_lin.weight = old_layer.weight
    new_lin.bias = old_layer.bias
    new_lin.in_features = old_layer.in_features
    new_lin.out_features = old_layer.out_features

    return new_lin


def set_quantize(model, quantize: bool = False, bitness: int = 4,
                 quantize_type: str = None,
                 trainable: bool = False, use_qloss: bool = False):
    for i in model.transformer.layer:
        i.ff.layer_1.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)
        i.ff.layer_2.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)


def clear_stat(model):
    for i in model.transformer.layer:
        i.ff.layer_1.quant.max_inp = []
        i.ff.layer_2.quant.max_weigh = []


def get_stat(model):
    for i in model.transformer.layer:
        print(i.ff.layer_1.quant.max_inp)
        print(i.ff.layer_1.quant.max_weigh)


def get_i_qsin_loss(model):
    res = 0

    for i in model.transformer.layer:
        input_loss = i.ff.layer_1.quant.q_i_sum + i.ff.layer_2.quant.q_i_sum
        res = res + input_loss

    return res


def get_w_qsin_loss(model):
    res = 0

    for i in model.transformer.layer:
        weight_loss = i.ff.layer_1.quant.q_w_sum + i.ff.layer_2.quant.q_w_sum
        res = res + weight_loss

    return res


def preprocess_data(task="cola", model="xlnet-base-cased"):
    # dataset = load_dataset("glue", task)
    # dataset.save_to_disk(f'./data/glue/data_{task}.pt')

    dataset = load_from_disk(f'./data/glue/data_{task}.pt')

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]
    # tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained('./tokenizers/xlnet-base-cased-tokenizer-glue', use_fast=False)

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key],
                             truncation=True, padding=True)

        return tokenizer(examples[sentence1_key], examples[sentence2_key],
                         truncation=True, padding=True)

    return dataset.map(preprocess_function, batched=True)


# def test_model(model)

def cola_run(task="cola", batch_size=16, num_bits=8):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = f'./model/model_{task}.pt'

    model_checkpoint = "xlnet-base-cased"
    encoded_dataset = preprocess_data(task, model_checkpoint)

    metric = load_metric('glue', task)
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

    # if task == "mrpc":
    #     num_eph = 5
    # elif task == "rte":
    #     num_eph = 10
    # elif task == "sst2":
    #     num_eph = 10
    # elif task == "cola":
    #     num_eph = 20
    # elif task == "stsb":
    #     num_eph = 5
    # elif task == "qqp":
    #     num_eph = 5
    # elif task == "qnli":
    #     num_eph = 5
    # elif task == "mnli":
    #     num_eph = 5

    num_eph = 3

    args = TrainingArguments(
        output_dir="D:/data/checkpoints/test-glue/",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_eph,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model=metric_name,
        disable_tqdm=True,
        evaluation_strategy="epoch",
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    # tokenizer.save_pretrained('./tokenizers/xlnet-base-cased-tokenizer-glue')
    tokenizer = AutoTokenizer.from_pretrained('./tokenizers/xlnet-base-cased-tokenizer-glue', use_fast=False)

    # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    # torch.save(model, f'./model/model_{task}_not_trained.pt')
    # model = torch.load(f'./model/model_{task}_not_trained.pt')
    #
    # for i in model.transformer.layer:
    #     i.ff.layer_1 = copy_lin_layer(i.ff.layer_1, 768, 3072)
    #     i.ff.layer_2 = copy_lin_layer(i.ff.layer_2, 3072, 768)

    # trainer = Trainer(
    #     model,
    #     args,
    #     train_dataset=encoded_dataset["train"],
    #     eval_dataset=encoded_dataset[validation_key],
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )
    # start_time = time.time()
    #
    # trainer.train()
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(trainer.evaluate())

    # torch.save(model, f'./model/model_{task}_trained.pt')





    model = torch.load(f'./model/model_{task}_trained.pt')

    model.to(device)
    encoded_dataset = preprocess_data(task, model_checkpoint)
    train_enc = encoded_dataset[validation_key]
    train_enc.set_format(type='torch', columns=['attention_mask', 'idx', 'input_ids', 'label', 'token_type_ids'])
    train_loader = DataLoader(train_enc, batch_size=16, shuffle=False)

    set_quantize(model, True, num_bits, "static_train")
    model.eval()

    start_time = time.time()

    for epoch, batch in zip(range(1), train_loader):
        print("!")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        res = model(input_ids, attention_mask=attention_mask)
        m = torch.nn.Softmax(dim=1)
        print("~~~~~~~~~~~~~~~~~")
        print([numpy.argmax(i.detach().cpu()).item() for i in m(res["logits"])])
        print(labels.tolist())
        r = [i == j for i, j in zip(labels.tolist(), [numpy.argmax(i.detach().cpu()).item() for i in m(res["logits"])])]
        print(r)
        print(sum(r) / len(r))

    torch.save(model, f'./model/model_{task}_stat_{num_bits}.pt')
    model = torch.load(f'./model/model_{task}_stat_{num_bits}.pt')

    set_quantize(model)

    print("--- %s seconds ---" % (time.time() - start_time))

    model = torch.load(f'./model/model_{task}_stat_{num_bits}.pt')
    encoded_dataset = preprocess_data(task, model_checkpoint)

    def continue_processing(num_bits, quant_type):

        set_quantize(model, True, num_bits, quant_type)

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        print(f"\n{task}\n\n{num_bits} bit, {quant_type}:\n")

        print(trainer.evaluate())

    continue_processing(num_bits, "dynamic")
    continue_processing(num_bits, "static")

    # model.to(device)
    # encoded_dataset = preprocess_data(task, model_checkpoint)
    # train_enc = encoded_dataset[validation_key]
    # train_enc.set_format(type='torch', columns=['attention_mask', 'idx', 'input_ids', 'label', 'token_type_ids'])
    # train_loader = DataLoader(train_enc, batch_size=16, shuffle=False)
    # model.eval()
    # set_quantize(model, True, num_bits, "static")
    # for epoch, batch in zip(range(1), train_loader):
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     labels = batch['label'].to(device)
    #     res = model(input_ids, attention_mask=attention_mask)
    #     m = torch.nn.Softmax(dim=1)
    #     print("~~~~~~~~~~~~~~~~~")
    #     print([numpy.argmax(i.detach().cpu()).item() for i in m(res["logits"])])
    #     print(labels.tolist())
    #     r = [i == j for i, j in zip(labels.tolist(), [numpy.argmax(i.detach().cpu()).item() for i in m(res["logits"])])]
    #     print(r)
    #     print(sum(r) / len(r))
    #
    # class qsinTrainer(Trainer):
    #     def compute_loss(self, model, inputs, return_outputs=False):
    #         """
    #         How the loss is computed by Trainer. By default, all models return the loss in the first element.
    #
    #         Subclass and override for custom behavior.
    #         """
    #         if self.label_smoother is not None and "labels" in inputs:
    #             labels = inputs.pop("labels")
    #         else:
    #             labels = None
    #         outputs = model(**inputs)
    #
    #         if self.args.past_index >= 0:
    #             self._past = outputs[self.args.past_index]
    #
    #         if labels is not None:
    #             loss = self.label_smoother(outputs, labels)
    #         else:
    #             # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #
    #         qsin_loss = get_i_qsin_loss(model) * 0.0000001
    #         qsin_loss = qsin_loss + get_w_qsin_loss(model) * 0.001
    #
    #         sum_loss = qsin_loss + loss
    #
    #         return (sum_loss, outputs) if return_outputs else sum_loss
    #
    # model = torch.load(f'./model/model_{task}_stat_{num_bits}.pt')
    #
    # def qsin_run(num_bits):
    #
    #     print(f"\n\nModel qsin {num_bits} bit train:\n\n")
    #
    #     model.train()
    #
    #     set_quantize(model, True, num_bits, "static", trainable=True, use_qloss=True)
    #
    #     trainer = qsinTrainer(
    #         model,
    #         args,
    #         train_dataset=encoded_dataset["train"],
    #         eval_dataset=encoded_dataset[validation_key],
    #         tokenizer=tokenizer,
    #         compute_metrics=compute_metrics
    #     )
    #
    #     start_time = time.time()
    #
    #     trainer.train()
    #
    #     trainer.evaluate()
    #
    #     print("--- %s seconds ---" % (time.time() - start_time))
    #
    #
    # qsin_run(num_bits)
    #
    # encoded_dataset = preprocess_data(task, model_checkpoint)
    # train_enc = encoded_dataset[validation_key]
    # train_enc.set_format(type='torch', columns=['attention_mask', 'idx', 'input_ids', 'label', 'token_type_ids'])
    # train_loader = DataLoader(train_enc, batch_size=16, shuffle=False)
    # model.eval()
    # set_quantize(model, True, num_bits, "static", trainable=True, use_qloss=True)
    # for epoch, batch in zip(range(1), train_loader):
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     labels = batch['label'].to(device)
    #     res = model(input_ids, attention_mask=attention_mask)
    #     m = torch.nn.Softmax(dim=1)
    #     print("~~~~~~~~~~~~~~~~~")
    #     print([numpy.argmax(i.detach().cpu()).item() for i in m(res["logits"])])
    #     print(labels.tolist())
    #     r = [i == j for i, j in zip(labels.tolist(), [numpy.argmax(i.detach().cpu()).item() for i in m(res["logits"])])]
    #     print(r)
    #     print(sum(r) / len(r))
    # # Дообучение
    #
    # model = torch.load(f'./model/model_{task}_stat_{num_bits}.pt')
    #
    # model.train()
    #
    # print("\nNo qsin\n")
    #
    # set_quantize(model, True, num_bits, "static", trainable=True, use_qloss=False)
    #
    # trainer = Trainer(
    #     model,
    #     args,
    #     train_dataset=encoded_dataset["train"],
    #     eval_dataset=encoded_dataset[validation_key],
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )
    #
    # start_time = time.time()
    #
    # trainer.train()
    # trainer.evaluate()
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
