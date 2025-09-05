We use [LongBench v1](https://github.com/THUDM/LongBench/tree/main/LongBench) to evaluate the trained model. And we delete the un-related files in the repo to avoiding confusing.  

Since the model has thinking output, we have made appropriate updates based on the basic retention of the original information of `dataset2maxlen.json` and `dataset2prompt.json`.

### Install
```shell
conda create -n longbench python=3.10  # with independent environment
conda activate longbench

cd CARE/evaluation/LongBench
pip3 install -r requirements.txt
```

Remind, if you want to test with official model, you should replace the config files with [official config](https://github.com/THUDM/LongBench/tree/main/LongBench/config)

### Evaluate
The datasets inference result  
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python pred.py --model Qwen2.5-7B-CARE   # replace with your one or multi cards id
```

The metric calculation from the `pred` folder.  
```shell
python3 eval.py --model Qwen2.5-7B-CARE
```
