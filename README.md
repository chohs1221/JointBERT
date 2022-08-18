# JointBERT

`Paper`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)   


## JointBERT Architecture

<p float="left" align="left">
    <img width="600" src="https://user-images.githubusercontent.com/56873395/185276547-c24c9531-574e-456f-9f4e-9fe1749f7725.png" />  
</p>


## JointBERT with POS tagging Architecture

<p float="left" align="left">
    <img width="600" src="https://user-images.githubusercontent.com/56873395/185415377-6df600bd-0ca2-484f-87f4-eccd5f5ccb6a.jpg" />  
</p>


## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |


## Training & Evaluation

```bash
$ python3 main.py --task {task_name} \
                  --epoch {epoch} \
                  --batch {batch_size} \

# For ATIS
$ python3 main.py --task atis \
                  --epoch 30 \
                  --batch 128 \
                  
# For Snips
$ python3 main.py --task snips \
                  --epoch 30 \
                  --batch 128 \
                  
# For JointBERT with POS tagging
$ python3 main_POS.py --task {task_name} \
                      --epoch {epoch} \
                      --batch {batch_size} \
```


## Results

- 30 epoch
- 128 batch size

|                      | **Snips**                                     ||| **ATIS**                                      |||
| -------------------- | :------------: | :---------: | :--------------: | :------------: | :---------: | :--------------: |
|                      | Intent acc (%) | Slot F1 (%) | Sentence acc (%) | Intent acc (%) | Slot F1 (%) | Sentence acc (%) | 
| JointBERT            | **98.4**       | **95.7**    | **90.1**         | **97.5**       | 94.4        | 85.1             | 
| JointBERT with POS   | 98.3           | 94.0        | 86.7             | 97.1           | **95.2**    | **86.3**         |


## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [monologg / JointBERT](https://github.com/monologg/JointBERT)
