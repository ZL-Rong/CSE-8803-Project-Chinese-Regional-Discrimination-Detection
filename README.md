The model we used is based on this work: https://github.com/ZL-Rong/Bert-Chinese-Text-Classification-Pytorch
Our work focus on Bert since it outperforms other models after tested them

Pretrained model download link  
bert_Chinese: pretrained model: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              corpus: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
from (https://github.com/huggingface/pytorch-transformers)   

Procedure:
After you downloaded the pretrained model
```
# train and test
python run.py --model bert

# predict
# place your predict.txt file into the THUCNews/data directory
python predict.py --model bert

# There will be a output.csv file containing input and output. 
、、、


## Related Paper
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] ERNIE: Enhanced Representation through Knowledge Integration  
