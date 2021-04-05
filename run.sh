#Train api
# python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 3 \
#   --max_seq_length 1000 \
#   --doc_stride 128 \
#   --output_dir /home/dev01/saurabh/token_classifier_r/output/bert/version_2 \
#   --cache_dir /home/dev01/saurabh/token_classifier_r/cache/bert/version_2 \
#   --logging_dir /home/dev01/saurabh/token_classifier_r/logs/bert/version_2 \
#   --save_total_limit 2

#   --train_file /home/dev01/saurabh/token_classifier_r/data/bert/version_1/train-v1.1.json \


python run_qa.py \
  --model_name_or_path  /home/dev01/saurabh/token_classifier_r/output/bert/version_2 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --cache_dir /home/dev01/saurabh/token_classifier_r/cache/bert/version_2 \
  --logging_dir /home/dev01/saurabh/token_classifier_r/logs/bert/version_2 \
  --do_predict \
  --test_file /home/dev01/saurabh/token_classifier_r/data/bert/version_1/test.json \
  --output_dir /home/dev01/saurabh/token_classifier_r/output/bert/version_2