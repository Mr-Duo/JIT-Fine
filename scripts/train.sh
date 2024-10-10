#!/bin/bash

python -m JITFine.concat.run \
    --output_dir=model/jitfine/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file dataset/FFmpeg/SETUP1/rus/SETUP1-FFmpeg-deepjit-train.jsonl dataset/FFmpeg/SETUP1/rus/SETUP1-FFmpeg-features-train.jsonl \
    --eval_data_file dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-val.jsonl dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-val.jsonl\
    --test_data_file dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-test.jsonl dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-test.jsonl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 128 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --feature_size 14 \
    --patience 10 \
    --seed 42 2>&1| tee model/jitfine/saved_models_concat/train.log
