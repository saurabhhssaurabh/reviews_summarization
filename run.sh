python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --train_file /home/dev01/saurabh/token_classifier_r/data/bert/version_1/train-v1.1.json \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /home/dev01/saurabh/token_classifier_r/output/bert/version_1

#   --do_eval \