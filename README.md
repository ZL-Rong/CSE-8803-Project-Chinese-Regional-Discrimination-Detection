# Results
Our team creates and releases the first-ever Chinese regional discrimination dataset that includes 500,000 comments from 12,000 posts posted by 300,000 users from Weibo, and proposed a Chinese regional discrimination BERT detection model. This large-scale dataset and the model allow us to detect regional discrimination speech in Chinese and to perform a rigorous measurement study characterizing the linguistic structure of regional discrimination speech for the first time.



# Procedures
The model we use is based on this work: https://github.com/ZL-Rong/Bert-Chinese-Text-Classification-Pytorch
Our work focus on Bert model since it outperforms other models after tested them

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
# place your predict.txt file with a default label value either 0 or 1 into the THUCNews/data directory
python predict.py --model bert

# There will be a output.csv file containing input and output. 
```

## Related Paper
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] ERNIE: Enhanced Representation through Knowledge Integration  
