from typing import Any
from torch import nn
import numpy
import torch
import torch.nn.functional as F
from transformers import Trainer


class KDTrainer(Trainer):

    def __init__(self, teacher=None, kd_loss_fn = 'kl', distill_hardness=0.5, distill_temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.distill_temperature = distill_temperature
        self.distill_hardness = distill_hardness
        self.kd_loss_fn = kd_loss_fn

        if self.teacher is not None:
            self.teacher.eval()
        self.criterion = torch.nn.CrossEntropyLoss()



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """

        student_outputs = model(**inputs)

        if self.teacher is None:
            loss = student_outputs["loss"]
        else:
            student_loss = student_outputs["loss"]
            input_device = inputs["input_ids"].device
            self.teacher = self.teacher.to(input_device)
            student_logits = student_outputs["logits"]
            labels = inputs["labels"]
            attention_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids,
                    attention_mask=attention_mask if attention_mask is not None else None,
                    labels = labels
                )
            teacher_logits =  teacher_outputs["logits"]
            student_logits_for_kd = student_logits
        
            if self.kd_loss_fn == 'kl':
                loss_func =  nn.KLDivLoss(reduction='batchmean')    
                kd_loss = loss_func(F.log_softmax(student_logits_for_kd / self.distill_temperature, dim=1),
                                        F.softmax(teacher_logits / self.distill_temperature, dim=1)) * self.distill_temperature ** 2
            elif self.kd_loss_fn == 'mse':
                loss_func = nn.MSELoss()
                kd_loss = loss_func(student_logits_for_kd, teacher_logits)

            else:
                raise('only support mse and kl loss')
            
            loss = ((1 - self.distill_hardness) * student_loss) + (self.distill_hardness * kd_loss)



        return (loss, student_outputs) if return_outputs else loss